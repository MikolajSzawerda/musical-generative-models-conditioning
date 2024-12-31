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

from audiocraft.data.audio import audio_write
import dataclasses
from pathlib import Path
from data import Concept, TextConcepts
from data_const import Datasets
from audioldm_eval.metrics.fad import FrechetAudioDistance
import logging

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


class GenEvalCallback(L.Callback):
    def __init__(
        self,
        fad: FrechetAudioDistance,
        base_dir: Datasets,
        cfg: EvaluationCallbackConfig,
    ):
        super().__init__()
        self.fad = fad
        self.cfg = cfg
        self.base_dir = base_dir.value

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

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.cfg.n_epochs != 0:
            return
        logger.info(f"Generation time at epoch {trainer.current_epoch + 1}")
        fads: list[float] = []

        def evaluate_concept(concept: Concept):
            logger.info("Started evaluating %s" % concept.name)
            results = pl_module.model.model.generate(
                [self.cfg.prompt_template % concept.pseudoword()]
                * self.cfg.n_generations
            )
            audio_list = []
            img_list = []
            for a_idx, music in enumerate(results):
                music = music.cpu()
                music = music / np.max(np.abs(music.numpy()))
                path = OUTPUT_PATH(
                    self.base_dir, concept.name, "temp", f"music_p{a_idx}"
                )
                audio_write(path, music, pl_module.model.model.cfg.sample_rate)
                audio_list.append(
                    wandb.Audio(
                        path + ".wav",
                        sample_rate=pl_module.model.model.cfg.sample_rate,
                        caption=f"{concept.name} audio {a_idx}",
                    )
                )
                if self.cfg.calc_spectrogram:
                    spectrogram = audio_to_spectrogram_image(
                        music.numpy(), pl_module.model.model.cfg.sample_rate
                    )
                    img_list.append(
                        wandb.Image(spectrogram, caption=f"Spectrogram {a_idx}")
                    )
            plots = {
                f"{concept.name}_audio": audio_list[:5],
                "global_step": trainer.global_step,
            }
            if self.cfg.calc_spectrogram:
                plots[f"{concept.name}_spectrogram"] = img_list
            pl_module.logger.experiment.log(plots)
            fad_score = self._calc_fad(concept.name)
            if fad_score != -1:
                pl_module.log(f"FAD {concept.name}", fad_score)
                fads.append(fad_score)
            else:
                logging.error("FAD %s RETURN -1" % concept.name)

        self.cfg.concepts.execute(evaluate_concept)
        if len(fads) > 0:
            pl_module.log(f"fad_avg", np.mean(fads))


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

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch % self.cfg.n_epochs != 0:
            return

        def update_best(concept: Concept):
            metrics = trainer.callback_metrics
            current_score = metrics.get(f"FAD {concept}")
            if current_score is None or current_score > self.best_score[concept.name]:
                return
            logger.info(
                f"Updating best saved embedings for {concept.name} at {trainer.current_epoch} epoch"
            )
            self.best_score[concept.name] = current_score.cpu().item()
            self.best_embeds[concept.name] = ConceptEmbeds(
                trainer.current_epoch,
                self.weights[concept.token_ids].detach().cpu(),
            )

        self.cfg.concepts.execute(update_best)
        wandb_logger = trainer.logger
        if isinstance(wandb_logger, WandbLogger):
            run_name = wandb_logger.experiment.name
        else:
            run_name = str(uuid.uuid4())
        save_file_path = MODELS_PATH(self.base_dir, f"{run_name}-best.pt")
        Path(MODELS_PATH(self.base_dir)).mkdir(parents=True, exist_ok=True)
        torch.save(self.best_embeds, save_file_path)
