from tools.project import INPUT_PATH
import torch
import os
from datasets import Dataset, DatasetDict
from datasets import load_dataset
from random import choice
from torch.utils.data import DataLoader, default_collate
from torch.utils.data import DataLoader
import pytorch_lightning as L
from data_const import train_desc, val_desc, Datasets
import dataclasses
from audiocraft.modules.conditioners import T5Conditioner
from audiocraft.models import MusicGen
import logging
from typing import Callable
from functools import cached_property
from toolz import concat

NUM_WORKERS = int(os.cpu_count() * 0.75)
logger = logging.getLogger(__name__)


def get_ds(dataset: Datasets) -> DatasetDict:
    return load_dataset(
        "json",
        data_files={
            "valid": INPUT_PATH(dataset.value, "metadata_val.json"),
            "train": INPUT_PATH(dataset.value, "metadata_train.json"),
        },
    )


def get_hg_ds():
    return load_dataset("mszawerd/concept-dataset")


class TokensProvider:
    def __init__(self, num: int):
        self.num = num

    def get(self, base: str):
        return [f"<{base}_{x}>" for x in range(self.num)]

    def get_str(self, base: str):
        return " ".join(self.get(base))


class PromptProvider:
    def __init__(self, prompts_template):
        self.template = prompts_template

    def get(self, *args):
        return choice(self.template) % args


@dataclasses.dataclass
class Concept:
    name: str
    token_ids: list[int]
    tokens: list[str]

    def pseudoword(self):
        return " ".join(self.tokens)


@dataclasses.dataclass
class ConceptEmbeds:
    epoch: int
    embeds: torch.Tensor


class TextConcepts:
    def __init__(self, concepts: list[Concept]):
        self.db: dict[str, Concept] = {c.name: c for c in concepts}

    @property
    def concepts(self) -> dict[str, Concept]:
        return self.db

    @cached_property
    def all_token_ids(self) -> list[int]:
        ids = []

        def collect_ids(c: Concept):
            ids.append(c.token_ids)

        self.execute(collect_ids)
        return list(concat(ids))

    def execute(self, func_by_concept: Callable[[Concept], None]):
        for concept in self.db.values():
            func_by_concept(concept)

    @classmethod
    def from_init(
        cls,
        text_conditioner: T5Conditioner,
        tokens_provider: TokensProvider,
        concepts: list[str],
    ):
        db: list[Concept] = []
        for concept in concepts:
            tokens = tokens_provider.get(concept)
            text_conditioner.t5_tokenizer.add_tokens(tokens)
            ids = text_conditioner.t5_tokenizer.convert_tokens_to_ids(tokens)
            db.append(Concept(concept, ids, tokens))
        text_conditioner.t5.resize_token_embeddings(len(text_conditioner.t5_tokenizer))
        return cls(db)

    @classmethod
    def from_musicgen(
        cls, music_model: MusicGen, tokens_provider: TokensProvider, concepts: list[str]
    ):
        return TextConcepts.from_init(
            list(music_model.lm.condition_provider.conditioners.values())[0],
            tokens_provider,
            concepts,
        )

    def __getitem__(self, item: str) -> Concept:
        return self.db[item]


class ConceptDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        ds: Dataset,
        split: str,
        concepts_db: TextConcepts,
        base_dir: Datasets,
        music_len: int = 100,
        pad_value: int = 0,
        preload_ds=True,
    ):
        self.ds = ds
        self.base_dir: str = INPUT_PATH(base_dir.value)

        self.concepts_db = concepts_db
        self.prompter = PromptProvider(val_desc if split == "valid" else train_desc)
        self.music_len = music_len
        self.split = split
        if preload_ds:
            self._preload_encoded(ds, pad_value)

    def _preload_encoded(self, ds: Dataset, pad_value: int = 0):
        logger.info("Started preloading audio embedings")
        enc_cache = [
            torch.load(os.path.join(self.base_dir, row["encoded_path"])).squeeze()
            for row in ds
        ]

        max_len = max(enc.shape[-1] for enc in enc_cache)

        def pad_encoding(enc: torch.Tensor, length: int) -> torch.Tensor:
            current_len = enc.shape[-1]
            if current_len >= length:
                return enc

            padded = enc.new_full((*enc.shape[:-1], length), pad_value)
            padded[..., :current_len] = enc
            return padded

        logger.info(f"Embedings padded to len: {max_len}")
        padded_list = [pad_encoding(enc, max_len) for enc in enc_cache]
        self.encoded_musics = torch.stack(padded_list, dim=0).contiguous()

    def __len__(self):
        return len(self.ds)

    def _random_slice(self, tensor):
        n, k = tensor.shape

        if self.music_len <= k:
            start_col = torch.randint(0, k - self.music_len + 1, (1,)).item()
            return tensor[:, start_col : start_col + self.music_len].detach()
        else:
            padding = torch.zeros(
                (n, self.music_len - k), device=tensor.device, dtype=torch.int64
            )
            return torch.cat((tensor, padding), dim=1).detach()

    def __getitem__(self, idx):
        row = self.ds[idx]
        concept = row["concept"]
        return {
            "encoded_music": self._random_slice(self.encoded_musics[idx]),
            "concept": concept,
            "prompt": self.prompter.get(self.concepts_db[concept].tokens),
            "new_tokens_ids": self.concepts_db[concept].token_ids,
        }


def collate_fn(batch):
    collated_batch = default_collate(batch)
    collated_batch["batch_tokens"] = torch.unique(
        torch.cat(collated_batch["new_tokens_ids"])
    )
    return collated_batch


class ConceptDataModule(L.LightningDataModule):
    def __init__(
        self,
        ds: DatasetDict,
        concepts_db: TextConcepts,
        base_dir: Datasets,
        music_len: int = 255,
        batch_size: int = 5,
        with_valid: bool = True,
    ):
        super().__init__()
        self.music_len = music_len
        self.batch_size = batch_size
        self.ds = ds
        self.concepts_db = concepts_db
        self.base_dir = base_dir
        self.with_valid = with_valid

    @classmethod
    def from_init(
        cls,
        concepts: list[str],
        files_base_dir: Datasets,
        ds: DatasetDict,
        text_conditioner: T5Conditioner,
        tokens_provider: TokensProvider,
        **kwargs,
    ):
        db = TextConcepts.from_init(text_conditioner, tokens_provider, concepts)
        return cls(ds, db, base_dir=files_base_dir, **kwargs)

    def setup(self, stage: str):
        self._train_ds = ConceptDataset(
            self.ds["train"],
            "train",
            self.concepts_db,
            self.base_dir,
            self.music_len,
        )
        if self.with_valid:
            self._val_ds = ConceptDataset(
                self.ds["valid"],
                "valid",
                self.concepts_db,
                self.base_dir,
                self.music_len,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=NUM_WORKERS,
            persistent_workers=True,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader | list:
        if not self.with_valid:
            return []
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            collate_fn=collate_fn,
            num_workers=NUM_WORKERS,
            persistent_workers=True,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    ds = get_ds(Datasets.TEXTUAL_INVERSION_V3).filter(
        lambda x: x["concept"] == "cluster_0"
    )
    conditioner = T5Conditioner("t5-small", 512, False, "cpu")
    cds = ConceptDataModule.from_init(
        ["cluster_0"],
        Datasets.TEXTUAL_INVERSION_V3,
        ds,
        conditioner,
        TokensProvider(10),
    )
    cds.setup("train")
    print(next(iter(cds.train_dataloader())))
    print(next(iter(cds.val_dataloader())))
