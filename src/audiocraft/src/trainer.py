import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from tools.project import LOGS_PATH
import torch
import wandb
from data import ConceptDataModule, get_ds
from argparse import ArgumentParser
import yaml
from audiocraft.models import MusicGen
from audioldm_eval.metrics.fad import FrechetAudioDistance
import logging
from data_const import Datasets
from model import ModelConfig, TransformerTextualInversion
from callbacks import (
    EmbedingsSaveCallbackConfig,
    EvaluationCallbackConfig,
    SaveEmbeddingsCallback,
    GenEvalCallback,
)

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WANDB_PROJECT = "debug"
SEED = 42

EXP_DATASET = Datasets.TEXTUAL_INVERSION_V3


def preprocess_ds(ds, concepts_ratio: float):
    shuffled = ds["train"].shuffle(seed=42)

    concept2indexes = {}
    concepts = shuffled["concept"]
    for i, c in enumerate(concepts):
        if c not in concept2indexes:
            concept2indexes[c] = []
        concept2indexes[c].append(i)

    keep_indexes = []
    for c, idxs in concept2indexes.items():
        n_keep = int(concepts_ratio * len(idxs))
        keep_indexes.extend(idxs[: min(100, len(idxs))])

    keep_indexes.sort()

    sampled_dataset = shuffled.select(keep_indexes)
    print(sampled_dataset.to_pandas().groupby("concept").size())
    ds["train"] = sampled_dataset
    return ds


def run_exp(cfg: ModelConfig, wandb_logger):
    logger.info("Loading MusicGen")
    model_name = f"facebook/musicgen-{cfg.model_name}"
    music_model = MusicGen.get_pretrained(model_name, device=DEVICE)
    music_model.set_generation_params(
        use_sampling=True, top_k=250, duration=cfg.examples_len, cfg_coef=cfg.cfg_coef
    )
    model = TransformerTextualInversion.from_musicgen(music_model, cfg)

    logger.info("Loading Dataset")
    ds = get_ds(EXP_DATASET).filter(lambda x: x["concept"] in cfg.concepts)
    dm = ConceptDataModule(
        ds,
        model.model.db,
        base_dir=EXP_DATASET,
        music_len=249,
        batch_size=cfg.batch_size,
    )

    fad = FrechetAudioDistance(verbose=True, use_pca=True, use_activation=True)

    quick_save_cl = SaveEmbeddingsCallback(
        EXP_DATASET,
        model.model.text_weights,
        EmbedingsSaveCallbackConfig(model.model.db),
    )
    early_stopping = L.callbacks.EarlyStopping(
        monitor="fad_avg", patience=331, mode="min", verbose=True
    )
    eval_cl = GenEvalCallback(
        fad,
        EXP_DATASET,
        EvaluationCallbackConfig(
            model.model.db,
            cfg.tokens_num,
        ),
    )
    trainer = L.Trainer(
        callbacks=[
            eval_cl,
            quick_save_cl,
            early_stopping,
        ],
        enable_checkpointing=False,
        logger=wandb_logger,
        log_every_n_steps=10,
        max_epochs=140,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
    )

    logger.info("Let it roll!")
    trainer.fit(model, dm)


def run_sweep_exp():
    wandb.init()
    run_exp(
        ModelConfig(**wandb.config.as_dict()),
        WandbLogger(project=WANDB_PROJECT, save_dir=LOGS_PATH),
    )
    wandb.finish()


def run_args_exp(args):
    wandb_logger = WandbLogger(project=WANDB_PROJECT, save_dir=LOGS_PATH)
    wandb_logger.experiment.config["batch_size"] = args.batch_size
    wandb_logger.experiment.config["examples_len"] = args.examples_len
    wandb_logger.experiment.config["tokens_num"] = args.tokens_num
    wandb_logger.experiment.config["model_name"] = (
        f"facebook/musicgen-{args.model_name}"
    )
    wandb_logger.experiment.config["concepts"] = args.concepts
    wandb_logger.experiment.config["lr"] = args.lr
    wandb_logger.experiment.config["ortho_alpha"] = args.ortho_alpha
    wandb_logger.experiment.config["entropy_alpha"] = args.entropy_alpha
    wandb_logger.experiment.config["grad_amplify"] = args.grad_amplify
    wandb_logger.experiment.config["cr_margin"] = args.cr_margin
    wandb_logger.experiment.config["cfg_coef"] = args.cfg_coef
    run_exp(args, wandb_logger)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    L.seed_everything(SEED, workers=True)
    init_parser = ArgumentParser(add_help=False)
    init_parser.add_argument("--use-sweep", action="store_true")
    init_args, _ = init_parser.parse_known_args()
    if init_args.use_sweep:
        with open(LOGS_PATH("sweep_config.yaml")) as f:
            sweep_config = yaml.safe_load(f)
        sweep_id = wandb.sweep(sweep=sweep_config, project=WANDB_PROJECT)
        wandb.agent(sweep_id, function=run_sweep_exp, count=20)
    else:
        parser = ArgumentParser(parents=[init_parser])
        parser.add_argument("--examples-len", type=int, default=5)
        parser.add_argument("--tokens-num", type=int, default=20)
        parser.add_argument("--batch-size", type=int, default=10)
        parser.add_argument("--grad-amplify", type=float, default=10.0)
        parser.add_argument("--entropy-alpha", type=float, default=1e1)
        parser.add_argument("--ortho-alpha", type=float, default=1e-2)
        parser.add_argument("--cr-margin", type=float, default=1.5)
        parser.add_argument("--cfg-coef", type=float, default=3.0)
        parser.add_argument("--lr", type=float, default=1e-1)
        parser.add_argument("--model-name", type=str, default="small")
        # parser.add_argument("--previous-run", type=str, default="")
        parser.add_argument("--concepts", nargs="+", default=["8bit"])
        run_args_exp(parser.parse_args())
