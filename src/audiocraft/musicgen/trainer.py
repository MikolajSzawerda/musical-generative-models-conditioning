import pytorch_lightning as L
from pytorch_lightning.loggers import WandbLogger
from tools.project import LOGS_PATH
import torch
import wandb
from data import ConceptDataModule, get_ds
from argparse import ArgumentParser
import yaml
from audiocraft.models import MusicGen
from fadtk.model_loader import CLAPLaionModel
from fadtk.fad import FrechetAudioDistance
import logging
from data_const import Datasets
from model import ModelConfig, TransformerTextualInversion
from callbacks import (
    EmbedingsSaveCallbackConfig,
    EvaluationCallbackConfig,
    SaveEmbeddingsCallback,
    GenEvalCallback,
    EMACallback
)
from utils import suppress_all_output

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# WANDB_PROJECT = "debug"
WANDB_PROJECT = "textual-inversion-lr"
SEED = 42

EXP_DATASET = Datasets.TEXTUAL_INVERSION_V3

def resample_ds(ds, max_examples_num: int):
    def resample_split(split: str):
        by_concept_count ={}
        idxs = []
        concepts = ds[split]["concept"]
        for i, c in enumerate(concepts):
            ctn = by_concept_count.get(c, 0)
            if ctn > max_examples_num:
                continue
            by_concept_count[c] = ctn + 1
            idxs.append(i)
        res_ds = ds[split].select(idxs)
        print(split, res_ds.to_pandas().groupby("concept").size())
        return res_ds
    ds['train'] = resample_split('train')
    ds['valid'] = resample_split('valid')
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
    ds = resample_ds(ds, cfg.examples_num)
    dm = ConceptDataModule(
        ds,
        model.model.db,
        base_dir=EXP_DATASET,
        music_len=249,
        batch_size=cfg.batch_size,
        randomize_tokens = cfg.randomize_tokens
    )
    with suppress_all_output():
        clap = CLAPLaionModel('music')
        fad = FrechetAudioDistance(clap)


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
        clap,
        EXP_DATASET,
        EvaluationCallbackConfig(
            model.model.db,
            cfg.tokens_num,
            randomize_tokens=cfg.randomize_tokens
        ),
    )
    trainer = L.Trainer(
        callbacks=[
            # EMACallback(0.05),
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
    wandb_logger.experiment.config["examples_num"] = args.examples_num
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
    wandb_logger.experiment.config["randomize_tokens"] = args.randomize_tokens
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
        wandb.agent(sweep_id, function=run_sweep_exp, count=12)
    else:
        parser = ArgumentParser(parents=[init_parser])
        parser.add_argument("--examples-len", type=int, default=5)
        parser.add_argument("--examples-num", type=int, default=100)
        parser.add_argument("--tokens-num", type=int, default=20)
        parser.add_argument("--batch-size", type=int, default=10)
        parser.add_argument("--grad-amplify", type=float, default=10.0)
        parser.add_argument("--entropy-alpha", type=float, default=1e0)
        parser.add_argument("--ortho-alpha", type=float, default=1e-1)
        parser.add_argument("--cr-margin", type=float, default=1.5)
        parser.add_argument("--cfg-coef", type=float, default=3.0)
        parser.add_argument("--lr", type=float, default=1e-1)
        parser.add_argument("--model-name", type=str, default="small")
        parser.add_argument("--randomize-tokens", dest='randomize_tokens', action='store_true')
        parser.add_argument("--no-randomize-tokens", dest='randomize_tokens', action='store_false')
        # parser.add_argument("--previous-run", type=str, default="")
        parser.add_argument("--concepts", nargs="+", default=["8bit"])
        run_args_exp(parser.parse_args())
