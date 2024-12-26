from util_tools import compute_cross_entropy, compute_ortho_loss
from model import append_new_tokens, TransformerTextualInversion, SaveEmbeddingsCallback, GenEvalCallback

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
from data import TokensProvider, ConceptDataModule, get_ds, TokensProvider
import uuid
from argparse import ArgumentParser
import yaml

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_read, audio_write
from audioldm_eval.metrics.fad import FrechetAudioDistance

DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
WANDB_PROJECT = "ti-debug"

def run_exp(cfg, wandb_logger):
    model_name = f"facebook/musicgen-{cfg.model}"
    ds = get_ds().filter(lambda x: x['concept'] in cfg.concepts)

    tokens_provider = TokensProvider(cfg.tokens_num)
    tokens_by_concept = {concept: list(tokens_provider.get(concept)) for concept in cfg.concepts}

    music_model = MusicGen.get_pretrained(model_name)
    music_model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=cfg.examples_len
    )
    text_conditioner=list(music_model.lm.condition_provider.conditioners.values())[0]
    tokenizer=text_conditioner.t5_tokenizer
    text_model=text_conditioner.t5

    tokens_ids_by_concept, tokens_ids = append_new_tokens(tokenizer, tokens_by_concept)
    text_model.resize_token_embeddings(len(tokenizer))

    fad = FrechetAudioDistance(verbose=True, use_pca=True, use_activation=True)
    dm = ConceptDataModule(ds, tokens_provider, tokens_ids_by_concept, music_len=249, batch_size=cfg.batch_size)
    model = TransformerTextualInversion(text_model, tokenizer, music_model, text_conditioner, tokens_ids, cfg.tokens_num, grad_amplify=cfg.grad_amp, lr=cfg.lr, ortho_alpha=cfg.ortho_alpha, entropy_alpha=cfg.entropy_alpha)

    quick_save_cl = SaveEmbeddingsCallback(MODELS_PATH('textual-inversion-v3'), cfg.concepts, tokens_ids_by_concept, text_model.shared.weight)
    early_stopping = L.callbacks.EarlyStopping(
        monitor="fad_avg",
        patience=250,
        mode="min",
        verbose=True
    )
    trainer = L.Trainer(callbacks=[GenEvalCallback(cfg.concepts, fad, cfg.tokens_num), quick_save_cl, early_stopping], enable_checkpointing=False, logger=wandb_logger, log_every_n_steps=10, max_epochs=250)
    trainer.fit(model, dm)


def run_sweep_exp():
    wandb.init()
    run_exp(wandb.config, WandbLogger(project=WANDB_PROJECT, save_dir=LOGS_PATH))
    wandb.finish()

def run_args_exp(args):
    logger = WandbLogger(project=WANDB_PROJECT, save_dir=LOGS_PATH)
    logger.experiment.config['batch_size'] = args.batch_size
    logger.experiment.config['examples_len'] = args.examples_len
    logger.experiment.config['tokens_num'] = args.tokens_num
    logger.experiment.config['model'] = f"facebook/musicgen-{args.model}"
    logger.experiment.config['concepts'] = args.concepts
    logger.experiment.config['lr'] = args.lr
    logger.experiment.config['ortho_alpha'] = args.ortho_alpha
    logger.experiment.config['entropy_alpha'] = args.entropy_alpha
    logger.experiment.config['grad_amp'] = args.grad_amp   
    run_exp(args, logger)

if __name__ == '__main__':
    init_parser = ArgumentParser(add_help=False)
    init_parser.add_argument("--use-sweep", action="store_true")
    init_args, _ = init_parser.parse_known_args()
    if init_args.use_sweep:
        with open(LOGS_PATH("sweep_config.yaml")) as f:
            sweep_config = yaml.safe_load(f)
        sweep_id = wandb.sweep(sweep=sweep_config, project=WANDB_PROJECT)
        wandb.agent(sweep_id, function=run_sweep_exp, count=5)
    else:
        parser = ArgumentParser(parents=[init_parser])
        parser.add_argument("--examples-len", type=int, default=5)
        parser.add_argument("--tokens-num", type=int, default=5)
        parser.add_argument("--batch-size", type=int, default=5)
        parser.add_argument("--grad-amp", type=float, default=10.0)
        parser.add_argument("--entropy-alpha", type=float, default=1e1)
        parser.add_argument("--ortho-alpha", type=float, default=1e-2)
        parser.add_argument("--lr", type=float, default=1e-1)
        parser.add_argument("--model", type=str, default="small")
        parser.add_argument("--concepts", nargs="+", default=["8bit"])
        run_args_exp(parser.parse_args())