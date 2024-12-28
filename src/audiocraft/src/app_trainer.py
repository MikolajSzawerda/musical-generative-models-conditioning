from util_tools import compute_cross_entropy, compute_ortho_loss
from model import append_new_tokens, TransformerTextualInversion

from torch.utils.data import DataLoader, default_collate
import tqdm
import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from datasets import load_dataset
from tools.project import INPUT_PATH, LOGS_PATH, OUTPUT_PATH, MODELS_PATH
import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import contextlib
import io
import os
import wandb
from data import (
    TokensProvider,
    ConceptDataModule,
    get_ds,
    TokensProvider,
    EvalDataModule,
)
import uuid
from argparse import ArgumentParser
import yaml
from datasets import Dataset, DatasetDict
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_read, audio_write
from audioldm_eval.metrics.fad import FrechetAudioDistance
from datasets import Audio, load_dataset

DATASET = "eval-ds"
WANDB_PROJECT = "eval-test"
SEED = 42

parser = ArgumentParser()
parser.add_argument("--examples-len", type=int, default=5)
parser.add_argument("--tokens-num", type=int, default=20)
parser.add_argument("--batch-size", type=int, default=10)
parser.add_argument("--grad-amp", type=float, default=10.0)
parser.add_argument("--entropy-alpha", type=float, default=1e1)
parser.add_argument("--ortho-alpha", type=float, default=1e-1)
parser.add_argument("--lr", type=float, default=1e-1)
parser.add_argument("--model", type=str, default="medium")
parser.add_argument("--previous-run", type=str, default="")
parser.add_argument("--concepts", nargs="+", default=["bells"])


class SaveEmbeddingsCallback(L.Callback):
    def __init__(
        self,
        save_path,
        concepts,
        tokens_ids_by_concept,
        tokens_num,
        weights,
        fad,
        n_epochs=10,
    ):
        super().__init__()
        self.save_path = save_path
        self.concepts = concepts
        self.best_score = {c: float("inf") for c in concepts}
        self.current_score = {c: float("inf") for c in concepts}
        self.best_file_path = None
        self.tokens_ids_by_concept = tokens_ids_by_concept
        self.weights = weights
        self.best_embeds = {
            c: weights[tokens_ids_by_concept[c]].detach().cpu() for c in concepts
        }
        self.n_epochs = n_epochs
        self.fad = fad
        self.tokens_num = tokens_num

    @torch.no_grad
    def calc_fad(self, pl_module, trainer):
        fads = []
        for concept in self.concepts:
            print(f"Generating: {concept}")
            response = pl_module.music_model.generate(
                [f"In the style of {TokensProvider(self.tokens_num).get_str(concept)}"]
                * 10
            )
            audio_list = []
            for a_idx in range(response.shape[0]):
                music = response[a_idx].cpu()
                music = music / np.max(np.abs(music.numpy()))
                path = OUTPUT_PATH(DATASET, concept, "temp", f"music_p{a_idx}")
                audio_write(path, music, pl_module.music_model.cfg.sample_rate)
                audio_wdb = wandb.Audio(
                    path + ".wav",
                    sample_rate=pl_module.music_model.cfg.sample_rate,
                    caption=f"{concept} audio {a_idx}",
                )
                audio_list.append(audio_wdb)

            pl_module.logger.experiment.log(
                {f"{concept}_audio": audio_list[:5], "global_step": trainer.global_step}
            )
            with contextlib.redirect_stdout(io.StringIO()):
                fd_score = self.fad.score(
                    INPUT_PATH(DATASET, "data", "train", f"{concept}", "fad"),
                    OUTPUT_PATH(DATASET, concept, "temp"),
                )
                os.remove(OUTPUT_PATH(DATASET, concept, "temp_fad_feature_cache.npy"))
                if isinstance(fd_score, int):
                    print("FAD RETURN -1")
                    continue
                val = list(fd_score.values())[0] * 1e-5
                self.current_score[concept] = val
                pl_module.log(f"FAD {concept}", val)
            fads.append(val)
        if len(fads) > 0:
            pl_module.log(f"fad_avg", np.mean(fads))

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch) % self.n_epochs == 0:
            self.calc_fad(pl_module, trainer)
            for concept in self.concepts:
                current_score = self.current_score[concept]
                if current_score is None or current_score > self.best_score[concept]:
                    continue
                print(
                    f"Updating best saved embedings for {concept} at {trainer.current_epoch} epoch"
                )
                self.best_score[concept] = current_score
                self.best_embeds[concept] = {
                    "epoch": trainer.current_epoch,
                    "embeds": self.weights[self.tokens_ids_by_concept[concept]]
                    .detach()
                    .cpu(),
                }

                wandb_logger = trainer.logger
                if isinstance(wandb_logger, WandbLogger):
                    run_name = wandb_logger.experiment.name
                else:
                    run_name = str(uuid.uuid4())
                save_file_path = os.path.join(self.save_path, f"{run_name}-best.pt")
                torch.save(self.best_embeds, save_file_path)


def run_training(cfg, wandb_logger):
    model_name = f"facebook/musicgen-{cfg.model}"
    ds = load_dataset(
        "json", data_files={"train": INPUT_PATH(DATASET, "metadata_train.json")}
    ).filter(lambda x: x["concept"] in cfg.concepts)
    tokens_provider = TokensProvider(cfg.tokens_num)
    tokens_by_concept = {
        concept: list(tokens_provider.get(concept)) for concept in cfg.concepts
    }

    music_model = MusicGen.get_pretrained(model_name)
    music_model.set_generation_params(
        use_sampling=True, top_k=250, duration=cfg.examples_len
    )
    text_conditioner = list(music_model.lm.condition_provider.conditioners.values())[0]
    tokenizer = text_conditioner.t5_tokenizer
    text_model = text_conditioner.t5

    tokens_ids_by_concept, tokens_ids = append_new_tokens(tokenizer, tokens_by_concept)
    text_model.resize_token_embeddings(len(tokenizer))

    fad = FrechetAudioDistance(verbose=True, use_pca=True, use_activation=True)
    dm = EvalDataModule(
        ds,
        tokens_provider,
        tokens_ids_by_concept,
        music_len=150,
        batch_size=cfg.batch_size,
        base_dir=DATASET,
    )

    if args.previous_run != "":
        previous_embeds = torch.load(
            MODELS_PATH(DATASET, f"{args.previous_run}-best.pt")
        )
    else:
        previous_embeds = None

    model = TransformerTextualInversion(
        text_model,
        tokenizer,
        music_model,
        text_conditioner,
        tokens_ids,
        tokens_ids_by_concept,
        cfg.tokens_num,
        grad_amplify=cfg.grad_amp,
        lr=cfg.lr,
        ortho_alpha=cfg.ortho_alpha,
        entropy_alpha=cfg.entropy_alpha,
        weights_by_concept=previous_embeds,
    )

    quick_save_cl = SaveEmbeddingsCallback(
        MODELS_PATH(DATASET),
        cfg.concepts,
        tokens_ids_by_concept,
        cfg.tokens_num,
        text_model.shared.weight,
        fad,
        n_epochs=30,
    )
    early_stopping = L.callbacks.EarlyStopping(
        monitor="fad_avg", patience=250, mode="min", verbose=True
    )
    trainer = L.Trainer(
        callbacks=[quick_save_cl],
        enable_checkpointing=False,
        logger=wandb_logger,
        log_every_n_steps=10,
        max_epochs=1000,
    )
    trainer.fit(model, dm)


if __name__ == "__main__":
    L.seed_everything(SEED, workers=True)
    args = parser.parse_args()
    logger = WandbLogger(project=WANDB_PROJECT, save_dir=LOGS_PATH)
    run_training(args, logger)
