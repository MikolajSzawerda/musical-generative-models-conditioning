import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from tools.project import INPUT_PATH, LOGS_PATH, OUTPUT_PATH, MODELS_PATH
import torch
import contextlib
import io
import os
import wandb
import uuid

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import json
from audiocraft.data.audio import audio_write
import dataclasses
from pathlib import Path
from data import Concept, TextConcepts
from data_const import Datasets
from fadtk.model_loader import CLAPLaionModel
from fadtk.fad import FrechetAudioDistance
import logging
from metrics import calc_fad, calc_clap
from utils import suppress_all_output
import shutil
import random
from toolz import partition_all, concat
import uuid
import tqdm
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ConceptEmbeds:
    epoch: int
    embeds: torch.Tensor


@dataclasses.dataclass
class EvaluationCallbackConfig:
    concepts: TextConcepts
    tokens_num: int
    n_epochs: int = 10
    n_generations: int = 10
    prompt_template: str = "In the style of %s"
    calc_spectrogram: bool = False
    generation_batch: int = 10
    generation_duration: int = 5


@dataclasses.dataclass
class EmbedingsSaveCallbackConfig:
    concepts: TextConcepts
    n_epochs: int = 10


def audio_to_spectrogram_image(audio, sr):
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
    def __init__(
        self,
        fad: FrechetAudioDistance,
        clap: CLAPLaionModel,
        base_dir: Datasets,
        cfg: EvaluationCallbackConfig,
    ):
        super().__init__()
        self.fad = fad
        self.clap = clap
        self.cfg = cfg
        self.base_dir = base_dir.value
        with open(INPUT_PATH(self.base_dir, "metadata_concepts.json"), "r") as fh:
            self.concept_descriptions = json.load(fh)

    def _calc_fad(self, concept: str):
        with contextlib.redirect_stdout(io.StringIO()):
            fd_score = self.fad.score(
                INPUT_PATH(self.base_dir, "data", "valid", f"{concept}", "fad"),
                OUTPUT_PATH(self.base_dir, concept, "temp"),
            )
            cache_path = OUTPUT_PATH(
                self.base_dir, concept, "temp_fad_feature_cache.npy"
            )
            if os.path.exists(cache_path):
                os.remove(cache_path)
            if isinstance(fd_score, int):
                return -1
            return list(fd_score.values())[0] * 1e-5
    
    @staticmethod
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def _calc_clap_score(self, concept: str, path: str):
        audio_embeds = self.fad.load_embeddings(path)
        text_embeds = self.clap.model.get_text_embedding(self.concept_descriptions[concept]).reshape(-1)
        return np.mean(self.cosine_similarity(audio_embeds, text_embeds))

    def _calc_fad_fadtk(self, concept: str):
        gen_path = OUTPUT_PATH(self.base_dir, concept, "temp")
        ref_path = INPUT_PATH(self.base_dir, "data", "train", concept, "audio")
        with suppress_all_output():
            for f in Path(gen_path).glob("*.*"):
                self.fad.cache_embedding_file(f)
            for f in Path(ref_path).glob("*.*"):
                self.fad.cache_embedding_file(f)
            score = self.fad.score(ref_path, gen_path)
            glob_score = self.fad.score('fma_pop', gen_path)
            clap_score = self._calc_clap_score(concept, gen_path)
            shutil.rmtree(os.path.join(gen_path, "embeddings"))
            shutil.rmtree(os.path.join(gen_path, "convert"))
            shutil.rmtree(os.path.join(gen_path, "stats"))
        return score, glob_score, clap_score

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.cfg.n_epochs != 0:
            return
        logger.info(f"Generation time at epoch {trainer.current_epoch + 1}")
        fads: list[float] = []
        ds_fads: list[float] = []
        claps: list[float] = []
        prompts = []

        def generate_prompts(concept: Concept):
            prompts.extend(
                [
                    (
                        concept.name,
                        self.cfg.prompt_template
                        % " ".join(random.sample(concept.tokens, len(concept.tokens))),
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
        pl_module.model.model.set_generation_params(duration=self.cfg.generation_duration)
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

        def calc_fad(concept: Concept):
            score, q_score, clap_score = self._calc_fad_fadtk(concept.name)
            pl_module.log(f"DS_FAD {concept.name}", score)
            pl_module.log(f"FAD {concept.name}", q_score)
            pl_module.log(f"CLAP {concept.name}", clap_score)
            fads.append(q_score)
            ds_fads.append(score)
            claps.append(clap_score)

        logger.info("Calculating FAD")
        self.cfg.concepts.execute(calc_fad)

        if len(fads) > 0:
            pl_module.log(f"fad_avg", np.mean(fads))
        if len(fads) > 0:
            pl_module.log(f"fad_ds_avg", np.mean(ds_fads))

        if len(claps) > 0:
            pl_module.log(f"clap_avg", np.mean(claps))


class SaveEmbeddingsCallback(L.Callback):
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

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.cfg.n_epochs != 0:
            return

        def update_best(concept: Concept):
            metrics = trainer.callback_metrics
            current_score = metrics.get(f"FAD {concept.name}")
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

        self.cfg.concepts.execute(update_best)
        wandb_logger = trainer.logger
        if isinstance(wandb_logger, WandbLogger):
            run_name = wandb_logger.experiment.name
        else:
            run_name = str(uuid.uuid4())
        save_file_path = MODELS_PATH(self.base_dir, f"{run_name}-best.pt")
        Path(MODELS_PATH(self.base_dir)).mkdir(parents=True, exist_ok=True)
        torch.save(self.best_embeds, save_file_path)
        values = self.best_score.values()
        valid_values = [x for x in values if x is not None and np.isfinite(x)]
        if len(valid_values) > 0:
            pl_module.log(f"fad_best_avg", np.mean(valid_values))
