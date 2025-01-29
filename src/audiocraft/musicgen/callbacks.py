import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from tools.project import INPUT_PATH, OUTPUT_PATH, MODELS_PATH
import torch
import os
import wandb

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import json
from audiocraft.data.audio import audio_write
import dataclasses
from .data import Concept, TextConcepts
from .data_const import Datasets
from fadtk.model_loader import CLAPLaionModel
from fadtk.fad import FrechetAudioDistance, calc_frechet_distance
from fadtk.utils import get_cache_embedding_path
import logging
from .utils import suppress_all_output
import shutil
import random
from toolz import partition_all
import uuid
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import tqdm

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class EvaluationCallbackConfig:
    """
    Configuration class for evaluation callbacks.

    Defines the parameters and settings required to configure evaluation callbacks,
    including concepts, token numbers, epochs, and various generation parameters.
    This class serves as a data container providing an organized structure for
    evaluation-related configurations in the application.

    :ivar concepts: Concepts for the evaluation process.
    :type concepts: TextConcepts
    :ivar tokens_num: Number of tokens to be considered for the evaluation.
    :type tokens_num: int
    :ivar n_epochs: Number of epochs for the evaluation. Defaults to 10.
    :type n_epochs: int
    :ivar n_generations: Number of generations for evaluation. Defaults to 10.
    :type n_generations: int
    :ivar prompt_template: Template for prompts provided during the evaluation.
        Default template is "In the style of %s".
    :type prompt_template: str
    :ivar calc_spectrogram: Boolean indicating whether to calculate spectrograms.
        Defaults to False.
    :type calc_spectrogram: bool
    :ivar generation_batch: Batch size for content generation.
        Defaults to 50.
    :type generation_batch: int
    :ivar generation_duration: Duration for each generation batch in seconds.
        Defaults to 5.
    :type generation_duration: int
    :ivar randomize_tokens: Boolean flag indicating whether to randomize tokens
        during evaluation. Defaults to True.
    :type randomize_tokens: bool
    """

    concepts: TextConcepts
    tokens_num: int
    n_epochs: int = 10
    n_generations: int = 10
    prompt_template: str = "In the style of %s"
    calc_spectrogram: bool = False
    generation_batch: int = 50
    generation_duration: int = 5
    randomize_tokens: bool = True


@dataclasses.dataclass
class EmbedingsSaveCallbackConfig:
    concepts: TextConcepts
    n_epochs: int = 10


def audio_to_spectrogram_image(audio, sr):
    """
    Convert an audio signal into a mel-spectrogram and represent it as an image. This function
    generates a mel-spectrogram from the input audio signal, visualizes it using a matplotlib
    plot, and then exports the visualization as a numpy array image. The resulting image can
    be used for various purposes, such as training machine learning models or analyzing audio
    data visually.

    :param audio: The input audio signal as a numpy array. If it is multi-dimensional, the
        function will squeeze it into a 1D array.
    :type audio: numpy.ndarray
    :param sr: The sampling rate of the audio signal.
    :type sr: int
    :return: A numpy array representing the mel-spectrogram image of the audio signal.
    :rtype: numpy.ndarray
    """
    if audio.ndim > 1:
        audio = np.squeeze(audio, axis=0)
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=sr / 2)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Plot the mel-spectrogram
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    librosa.display.specshow(
        S_dB, x_axis="time", y_axis="mel", sr=sr, fmax=sr / 2, ax=ax, cmap="magma"
    )
    ax.set_title("Mel-Spectrogram")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mel Frequency")

    # Convert plot to numpy array
    fig.canvas.draw()
    spectrogram_image = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)
    return spectrogram_image


from pathlib import Path


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def _calc_clap_score(fad, clap, concept: str, path: str, descriptions: dict[str, str]):
    audio_embeds = fad.load_embeddings(path)
    text_embeds = clap.model.get_text_embedding(descriptions[concept]).reshape(-1)
    return np.mean(cosine_similarity(audio_embeds, text_embeds))


@torch.no_grad()
def offline_eval(
    fad, clap, base_dir: str, concepts: list[str], descriptions: dict[str, str]
):
    """
    Evaluate the performance of the audio generation pipeline using various metrics such
    as Frechet Audio Distance (FAD) and Contrastive Language-Audio Pretraining (CLAP) score.

    This function evaluates the generated audio files against reference audio files. It
    makes use of precomputed background statistics and calculates performance metrics for
    each specified concept in the dataset. The function also caches audio embeddings for
    later evaluations and cleans temporary files after the process is complete.

    :param fad: Instance of a class that provides methods for loading and caching audio
        embeddings, as well as for loading precomputed statistics. Used for calculating
        FAD scores.
    :param clap: Instance of a class that provides methods needed for contrastive language-
        audio scoring (CLAP). Used to compute the CLAP score for generated audio files.
    :param base_dir: Base directory containing input and output data. This serves as
        the root directory for paths to generated and reference audio files.
    :param concepts: List of concept names (strings) to evaluate. Each concept corresponds
        to a specific category or subset of audio data for evaluation.
    :param descriptions: Dictionary where keys are concept names and values are their
        respective textual descriptions. Used during the computation of CLAP scores.
    :return: A dictionary where each key represents a concept name, and the corresponding
        value is another dictionary. That dictionary contains the metrics: Frechet Audio
        Distance (FAD) score, global FAD score, and the CLAP score.
    """
    res = {}
    mu_bg, cov_bg = fad.load_stats("fma_pop")
    for concept in concepts:
        gen_path = OUTPUT_PATH(base_dir, concept, "temp")
        ref_path = INPUT_PATH(base_dir, "data", "train", concept, "audio")

        def cache_path_emb(path: str):
            files = Path(path).glob("*.*")
            files = [
                f for f in files if not get_cache_embedding_path(clap.name, f).exists()
            ]
            if len(files) == 0:
                return
            for f in files:
                fad.cache_embedding_file(f)

        cache_path_emb(gen_path)
        cache_path_emb(ref_path)
        mu_gen, cov_gen = fad.load_stats(gen_path)
        mu_ref, cov_ref = fad.load_stats(ref_path)
        score = calc_frechet_distance(mu_ref, cov_ref, mu_gen, cov_gen)
        glob_score = calc_frechet_distance(mu_bg, cov_bg, mu_gen, cov_gen)
        clap_score = _calc_clap_score(fad, clap, concept, gen_path, descriptions)
        res[concept] = {"ds_fad": score, "fad": glob_score, "clap": clap_score}
        shutil.rmtree(os.path.join(gen_path, "embeddings"))
        shutil.rmtree(os.path.join(gen_path, "convert"))
        shutil.rmtree(os.path.join(gen_path, "stats"))
    return res


import sys


def _suppress_output():
    devnull = open(os.devnull, "w")
    sys.stdout = devnull
    sys.stderr = devnull


def calc_eval(
    base_dir: str, concepts: list[str], descriptions: dict[str, str], workers=2
):
    concepts_batches = list(partition_all(len(concepts) // workers, concepts))
    multiprocessing.set_start_method("spawn", force=True)
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_suppress_output
    ) as executor:
        results = list(
            executor.map(
                offline_eval,
                [base_dir] * len(concepts_batches),
                concepts_batches,
                [{k: descriptions[k] for k in batch} for batch in concepts_batches],
            )
        )

    return {k: v for val in results for k, v in val.items()}


class EMACallback(L.Callback):
    def __init__(self, decay=0.9999):
        super().__init__()
        self.decay = decay
        self.ema_weights = None
        self.old_weights = None

    def on_train_start(self, trainer, pl_module):
        ids = pl_module.model.db.all_token_ids
        self.ema_weights = pl_module.model.text_weights[ids].clone()

    @torch.no_grad()
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        ids = pl_module.model.db.all_token_ids
        self.ema_weights = (
            self.decay * self.ema_weights
            + (1 - self.decay) * pl_module.model.text_weights[ids].clone()
        )

    def apply_ema(self, pl_module):
        if self.ema_weights is None:
            return
        logger.info("EMA applied")
        ids = pl_module.model.db.all_token_ids
        old_embeds = pl_module.model.text_weights[ids].clone()
        self.old_weights = old_embeds
        pl_module.model.text_weights[ids] = self.ema_weights
        return old_embeds

    def remove_ema(self, pl_module):
        if self.old_weights is None:
            return
        logger.info("EMA removed")
        ids = pl_module.model.db.all_token_ids
        pl_module.model.text_weights[ids] = self.old_weights


class GenEvalCallback(L.Callback):
    """
    Provides functionality for evaluating generative models through metrics like FAD (Frechet Audio
    Distance), CLAP, and spectrogram visualizations by using a configurable callback. Executes
    evaluations periodically during training, based on the number of epochs and configuration settings.

    This class manages the evaluation process by generating prompts from predefined concepts,
    producing corresponding audio samples, computing evaluation metrics, and logging the
    results. It supports operations such as randomized token generation for prompts, audio and
    spectrogram generation, and offline evaluation with metrics like FAD and CLAP.

    :ivar cfg: Configuration for evaluation callback, including parameters like number of
        generations, epoch interval, and data structure details.
    :type cfg: EvaluationCallbackConfig
    :ivar fad: An instance of FrechetAudioDistance for calculating FAD metrics.
    :type fad: FrechetAudioDistance
    :ivar clap: An instance of CLAPLaionModel used for calculating CLAP scores.
    :type clap: CLAPLaionModel
    :ivar base_dir: Root directory containing dataset files and metadata for evaluation.
    :type base_dir: str
    :ivar concept_descriptions: A mapping of concepts to their descriptive metadata loaded
        from a JSON file.
    :type concept_descriptions: dict
    """

    def __init__(
        self,
        fad: FrechetAudioDistance,
        clap: CLAPLaionModel,
        base_dir: Datasets,
        cfg: EvaluationCallbackConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.fad = fad
        self.clap = clap
        self.base_dir = base_dir.value
        with open(INPUT_PATH(self.base_dir, "metadata_concepts.json"), "r") as fh:
            self.concept_descriptions = json.load(fh)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.cfg.n_epochs != 0:
            return
        logger.info(f"Generation time at epoch {trainer.current_epoch + 1}")
        fads: list[float] = []
        ds_fads: list[float] = []
        claps: list[float] = []
        prompts = []

        def generate_prompts(concept: Concept):
            if self.cfg.randomize_tokens:
                tokens = " ".join(random.sample(concept.tokens, len(concept.tokens)))
            else:
                tokens = concept.pseudoword()
            prompts.extend(
                [
                    (
                        concept.name,
                        self.cfg.prompt_template % tokens,
                    )
                    for _ in range(self.cfg.n_generations)
                ]
            )

        self.cfg.concepts.execute(generate_prompts)

        prompts_batches = partition_all(self.cfg.generation_batch, prompts)
        audio_list = {}
        img_list = []
        concept_counter = {}

        def save_audio(audio: torch.Tensor, concept_name: str):
            ctn = concept_counter.get(concept_name, 0)
            path = OUTPUT_PATH(self.base_dir, concept_name, "temp", f"music_p{ctn}")
            audio_write(path, audio, pl_module.model.model.cfg.sample_rate)
            concept_audio = audio_list.get(concept_name, [])
            concept_audio.append(
                wandb.Audio(
                    path + ".wav",
                    sample_rate=pl_module.model.model.cfg.sample_rate,
                    caption=f"{concept_name} audio {ctn}",
                )
            )
            audio_list[concept_name] = concept_audio

            if self.cfg.calc_spectrogram:
                spectrogram = audio_to_spectrogram_image(
                    audio.numpy(), pl_module.model.model.cfg.sample_rate
                )
                img_list.append(wandb.Image(spectrogram, caption=f"Spectrogram {ctn}"))

            concept_counter[concept_name] = ctn + 1

        old_duration = pl_module.model.model.duration
        pl_module.model.model.set_generation_params(
            duration=self.cfg.generation_duration
        )
        for prompts_batch in tqdm.tqdm(prompts_batches):
            concepts, prompts = list(zip(*prompts_batch))
            with torch.no_grad():
                results = pl_module.model.model.generate(prompts).cpu()
            results = results / np.max(np.abs(results.numpy()))
            for concept_name, audio in zip(concepts, results):
                save_audio(audio, concept_name)
        pl_module.model.model.set_generation_params(duration=old_duration)

        plots = {
            f"{concept_name}_audio": concept_audio[:5]
            for concept_name, concept_audio in audio_list.items()
        }
        plots.update(
            {
                "global_step": trainer.global_step,
            }
        )

        pl_module.logger.experiment.log(plots)
        logger.info("Started evalation")
        with suppress_all_output():
            eval_res = offline_eval(
                self.fad,
                self.clap,
                self.base_dir,
                self.cfg.concepts.concepts_names,
                self.concept_descriptions,
            )
        for c_name, stats in eval_res.items():
            pl_module.log(f"DS_FAD {c_name}", stats["ds_fad"])
            pl_module.log(f"FAD {c_name}", stats["fad"])
            pl_module.log(f"CLAP {c_name}", stats["clap"])
            ds_fads.append(stats["ds_fad"])
            fads.append(stats["fad"])
            claps.append(stats["clap"])

        if len(fads) > 0:
            pl_module.log(f"fad_avg", np.mean(fads))
        if len(fads) > 0:
            pl_module.log(f"fad_ds_avg", np.mean(ds_fads))

        if len(claps) > 0:
            pl_module.log(f"clap_avg", np.mean(claps))


class SaveEmbeddingsCallback(L.Callback):
    """
    SaveEmbeddingsCallback class.

    This class is a callback designed to save embeddings during the training process.
    It tracks the best embeddings associated with specific concepts based on evaluation
    metrics and stores them. The class enables logging and saving of embeddings in a
    systematic manner to aid in model training and analysis.

    It is intended to work with concepts configured in an external configuration object.
    Embeddings are updated and logged at the end of specific validation epochs based on
    performance improvements.

    :ivar base_dir: Base directory where embeddings files will be stored.
    :type base_dir: str
    :ivar cfg: Configuration object containing settings and concept information for saving embeddings.
    :type cfg: EmbedingsSaveCallbackConfig
    :ivar best_score: Dictionary maintaining the best evaluation scores for each concept seen during training.
    :type best_score: dict
    :ivar best_file_path: Path to the file saving the best embeddings.
    :type best_file_path: str or None
    :ivar weights: Tensor holding embedding information for all concepts.
    :type weights: torch.Tensor
    :ivar best_embeds: Dictionary maintaining the best embeddings for each concept.
    :type best_embeds: dict
    :ivar all_embeds: Dictionary maintaining all saved embeddings for every tracked epoch.
    :type all_embeds: dict
    """

    def __init__(
        self,
        base_dir: Datasets,
        weights: torch.Tensor,
        cfg: EmbedingsSaveCallbackConfig,
    ):
        super().__init__()
        self.base_dir = base_dir.value
        self.cfg = cfg
        self.best_score = {c: float("inf") for c in cfg.concepts.concepts.keys()}
        self.best_file_path = None
        self.weights = weights
        self.best_embeds = {
            c.name: weights[c.token_ids].detach().cpu()
            for c in cfg.concepts.concepts.values()
        }
        self.all_embeds = {}

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.cfg.n_epochs != 0:
            return

        def update_best(concept: Concept):
            metrics = trainer.callback_metrics
            current_score = metrics.get(f"DS_FAD {concept.name}")
            if current_score is None or current_score > self.best_score[concept.name]:
                return
            logger.info(
                f"Updating best saved embedings for {concept.name} at {trainer.current_epoch} epoch"
            )
            self.best_score[concept.name] = current_score.cpu().item()
            self.best_embeds[concept.name] = {
                "epoch": trainer.current_epoch,
                "embeds": self.weights[concept.token_ids].detach().cpu(),
            }

        def append_concept(concept: Concept):
            key = str(trainer.current_epoch)
            epoch_concepts = self.all_embeds.get(key, {})
            epoch_concepts[concept.name] = {
                "epoch": trainer.current_epoch,
                "embeds": self.weights[concept.token_ids].detach().cpu(),
            }
            self.all_embeds[key] = epoch_concepts

        self.cfg.concepts.execute(update_best)
        self.cfg.concepts.execute(append_concept)
        wandb_logger = trainer.logger
        if isinstance(wandb_logger, WandbLogger):
            run_name = wandb_logger.experiment.name
        else:
            run_name = str(uuid.uuid4())
        save_file_path = MODELS_PATH(self.base_dir, f"{run_name}-best.pt")
        Path(MODELS_PATH(self.base_dir)).mkdir(parents=True, exist_ok=True)
        torch.save(self.best_embeds, save_file_path)
        torch.save(self.all_embeds, MODELS_PATH(self.base_dir, f"{run_name}-all.pt"))
        values = self.best_score.values()
        valid_values = [x for x in values if x is not None and np.isfinite(x)]
        if len(valid_values) > 0:
            pl_module.log(f"fad_best_avg", np.mean(valid_values))


if __name__ == "__main__":
    import timeit

    with open(INPUT_PATH("concepts-dataset", "metadata_concepts.json"), "r") as fh:
        concept_descriptions = json.load(fh)

    def func1():
        calc_eval(
            "concepts-dataset",
            list(concept_descriptions.keys()),
            concept_descriptions,
            workers=4,
        )

    def func2():
        offline_eval(
            "concepts-dataset", list(concept_descriptions.keys()), concept_descriptions
        )

    time_1 = timeit.timeit(func1, number=1)
    time_2 = timeit.timeit(func2, number=1)
    print(time_1, time_2)
