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

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_read, audio_write
from audioldm_eval.metrics.fad import FrechetAudioDistance

parser = ArgumentParser()

parser.add_argument("--examples-len", type=int, default=5)
parser.add_argument("--tokens-num", type=int, default=5)
parser.add_argument("--batch-size", type=int, default=5)
parser.add_argument("--grad-amp", type=float, default=10.0)
parser.add_argument("--entropy-alpha", type=float, default=1e1)
parser.add_argument("--ortho-alpha", type=float, default=1e-2)
parser.add_argument("--lr", type=float, default=1e-1)
parser.add_argument("--model", type=str, default="small")
parser.add_argument("--concepts", nargs="+", default=["8bit"])

DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    args = parser.parse_args()
    model = f"facebook/musicgen-{args.model}"
    ds = get_ds().filter(lambda x: x['concept'] in args.concepts)

    tokens_provider = TokensProvider(args.tokens_num)
    tokens_by_concept = {concept: list(tokens_provider.get(concept)) for concept in args.concepts}

    music_model = MusicGen.get_pretrained(model)
    music_model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=args.examples_len
    )
    text_conditioner=list(music_model.lm.condition_provider.conditioners.values())[0]
    tokenizer=text_conditioner.t5_tokenizer
    text_model=text_conditioner.t5

    tokens_ids_by_concept, tokens_ids = append_new_tokens(tokenizer, tokens_by_concept)
    text_model.resize_token_embeddings(len(tokenizer))

    fad = FrechetAudioDistance(verbose=True, use_pca=True, use_activation=True)
    dm = ConceptDataModule(ds, tokens_provider, tokens_ids_by_concept, music_len=255, batch_size=args.batch_size)
    model = TransformerTextualInversion(text_model, tokenizer, music_model, text_conditioner, tokens_ids, grad_amplify=args.grad_amp, lr=args.lr, ortho_alpha=args.ortho_alpha, entropy_alpha=args.entropy_alpha)
    # tb_logger = L.loggers.TensorBoardLogger(LOGS_PATH, name='textual-inversion-v3')
    wandb_logger = WandbLogger(project='textual-inversion-v3', save_dir=LOGS_PATH)
    wandb_logger.experiment.config['batch_size'] = args.batch_size
    wandb_logger.experiment.config['examples_len'] = args.examples_len
    wandb_logger.experiment.config['tokens_num'] = args.tokens_num
    wandb_logger.experiment.config['model'] = model
    wandb_logger.experiment.config['concepts'] = args.concepts
    wandb_logger.experiment.config['lr'] = args.lr
    wandb_logger.experiment.config['ortho_alpha'] = args.ortho_alpha
    wandb_logger.experiment.config['entropy_alpha'] = args.entropy_alpha
    wandb_logger.experiment.config['grad_amp'] = args.grad_amp

    quick_save_cl = SaveEmbeddingsCallback(LOGS_PATH('embeds'), args.concepts, tokens_ids_by_concept, text_model.shared.weight)
    trainer = L.Trainer(callbacks=[GenEvalCallback(args.concepts, fad), quick_save_cl], enable_checkpointing=False, logger=wandb_logger, log_every_n_steps=10)
    trainer.fit(model, dm)