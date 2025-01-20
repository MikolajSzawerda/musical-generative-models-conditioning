import random


from data_const import train_desc, val_desc
import pytorch_lightning as L
import os
from pytorch_lightning.loggers import WandbLogger

import torch.nn as nn
import torch.nn.functional as F
from audiocraft.models import MusicGen
import torch
from tools.project import INPUT_PATH, LOGS_PATH
from data import TextConcepts, TokensProvider, Concept
from losses import compute_cross_entropy, compute_ortho_loss
from model import TIMusicGen, ModelConfig
from torch.optim import Adam
from torch.utils.data import (
    Dataset,
    DataLoader,
    TensorDataset,
    random_split,
    ConcatDataset,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = int(os.cpu_count() * 0.75)
SEED = 42


class ConceptTensorDataset(Dataset):
    def __init__(self, split: str, concept: Concept):
        ds = torch.load(
            INPUT_PATH("cifar", f"encoded_{concept.name}.pt"),
            map_location="cpu",
        )[:225, :, :].cpu()
        ds = TensorDataset(ds)
        train_ds, val_ds = random_split(
            ds, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
        )
        self.split = split
        if self.split == "train":
            self.ds = train_ds
            self.desc = train_desc
        else:
            self.ds = val_ds
            self.desc = val_desc

        self.concept = concept

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        true_idx = idx
        row = self.ds[true_idx]
        return {
            "encoded_music": row[0],
            "concept": self.concept.name,
            "prompt": "In the style of %s" % self.concept.pseudoword(),
            "new_tokens_ids": self.concept.token_ids,
        }


class ConceptClipDataset(Dataset):
    def __init__(self, img_ds: TensorDataset, concept_ds: ConceptTensorDataset):
        self.img_ds = img_ds
        self.concept_ds = concept_ds

    def __len__(self):
        return len(self.img_ds) // 4

    def __getitem__(self, idx):
        img, label = self.img_ds[idx]
        m_idx = random.randint(0, len(self.concept_ds) - 1)
        m_data = self.concept_ds[m_idx]
        return {
            "img": img,
            "encoded_music": m_data["encoded_music"],
            "prompt": m_data["prompt"],
            "label": int(label == 3),
        }


class ConceptDataModule(L.LightningDataModule):
    def __init__(self, concepts_db: TextConcepts, batch_size: int = 10):
        super().__init__()
        self.concepts_db = concepts_db
        self.batch_size = batch_size

    def get_ds(self, concept: str, img_cls: str, split: str):
        music_ds = ConceptTensorDataset(split, self.concepts_db.concepts[concept])
        tr = torch.load(INPUT_PATH("cifar", f"{img_cls}_{split}_embeds.pt"))
        img_ds = TensorDataset(tr[0], tr[1])
        return ConceptClipDataset(img_ds, music_ds)

    def setup(self, stage: str):
        self._train_ds = ConcatDataset(
            [self.get_ds("8bit", "cat", "train"), self.get_ds("metal", "dog", "train")]
        )
        self._val_ds = ConcatDataset(
            [self.get_ds("8bit", "cat", "val"), self.get_ds("metal", "dog", "val")]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            persistent_workers=True,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader | list:
        return DataLoader(
            self._val_ds,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            persistent_workers=True,
        )


class FiLM(nn.Module):
    def __init__(self, feature_dim: int, cond_dim: int):
        super().__init__()
        self.layer_1 = nn.Linear(cond_dim, feature_dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        film_params = self.layer_1(cond)
        gamma, beta = film_params.chunk(2, dim=-1)
        return gamma * x + beta

class FiLMModule(nn.Module):
    def __init__(self, in_dim:int, out_dim: int, cond_dim: int, dropout_p: float=0.3):
        super().__init__()
        self.layer_1 = nn.Linear(in_dim, out_dim)
        self.film = FiLM(out_dim, cond_dim)
        self.dropout = nn.Dropout(dropout_p)


    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.layer_1(x)
        x = self.film(x, cond)
        x = self.dropout(x)
        return F.relu(x)

class ClipProjector(nn.Module):
    def __init__(
        self,
        tokens_num: int,
        n_classes: int,
        input_dim: int = 64,
        hidden_dim: int = 1024,
        cond_dim: int = 32,
        dropout_p: float = 0.3,
    ):
        super().__init__()
        self.tokens_num = tokens_num
        self.output_dim = tokens_num * 768

        self.class_embedding = nn.Embedding(
            num_embeddings=n_classes, embedding_dim=cond_dim
        )
        self.nn_1 = FiLMModule(input_dim, hidden_dim, cond_dim, dropout_p)
        self.nn_2 = FiLMModule(hidden_dim, hidden_dim, cond_dim, dropout_p)
        self.last_hidden_layer = nn.Linear(hidden_dim, self.output_dim)

    
    def init_last_layer(self, init_bias: torch.Tensor):
        with torch.no_grad():
            self.last_hidden_layer.bias.copy_(init_bias.unsqueeze(0).expand(self.tokens_num, -1).contiguous().view(-1))

    def forward(self, x: torch.Tensor, class_idx: torch.Tensor) -> torch.Tensor:
        cond = self.class_embedding(class_idx)

        x = self.nn_1(x, cond)
        x_2 = self.nn_2(x, cond)
        x = x + x_2
        x = self.last_hidden_layer(x)
        x = x.view(-1, self.tokens_num, 768)
        return x


class ClipTextualInversion(L.LightningModule):
    def __init__(
        self, projector: ClipProjector, music_model: TIMusicGen, cfg: ModelConfig
    ):
        super().__init__()
        self.projector = projector
        self.music_model = music_model
        self.cfg = cfg

    def _init_text_model(self):
        def zero_existing_emb(grad):
            mask = torch.zeros_like(grad)
            for new_token_id in self.music_model.db.all_token_ids:
                mask[new_token_id] = self.cfg.grad_amplify
            return grad * mask

        self.music_model.text_weights.register_hook(zero_existing_emb)

    def on_train_start(self):
        self.music_model.init_model_random()
        with torch.no_grad():
            self.projector.init_last_layer(self.music_model.text_model.shared.weight.mean(dim=0))
        self.music_model.enable_grad()
        self._init_text_model()
        

    def forward(self, img, music, prompt, label):
        tokenized = self.music_model.tokenizer(
            prompt, return_tensors="pt", padding=True, add_special_tokens=False
        )
        tokenized = {k: v.to(DEVICE) for k, v in tokenized.items()}

        mask = tokenized["attention_mask"]
        text_with_clip = self.music_model.text_model.shared.weight[
            tokenized["input_ids"]
        ]
        # HARDOCODED AS PROMPT IS HARDOCODED
        img_projection = self.projector(img, label)
        text_with_clip[:, -self.cfg.tokens_num :, :] = img_projection
        with self.music_model.text_conditioner.autocast and torch.set_grad_enabled(
            True
        ):
            text_emb = self.music_model.text_model(
                inputs_embeds=text_with_clip, attention_mask=mask
            ).last_hidden_state
        text_emb = self.music_model.text_conditioner.output_proj(
            text_emb.to(self.music_model.text_conditioner.output_proj.weight)
        )
        text_emb = text_emb * mask.unsqueeze(-1)
        with self.music_model.model.autocast:
            return (
                self.music_model.model.lm.compute_predictions(
                    music, [], {"description": (text_emb, mask)}
                ),
                img_projection,
            )

    def training_step(self, batch, batch_idx):
        self.music_model.enable_grad()

        img, music, prompt = batch["img"], batch["encoded_music"], batch["prompt"]
        out, proj = self(img, music, prompt, batch["label"])
        loss, _ = compute_cross_entropy(out.logits, music, out.mask)
        ortho_loss = torch.vmap(compute_ortho_loss, in_dims=0)(proj).mean()
        loss += self.cfg.ortho_alpha * ortho_loss
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("ortho_loss", ortho_loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss * 100.0

    def validation_step(self, batch, batch_idx):
        img, music, prompt = batch["img"], batch["encoded_music"], batch["prompt"]
        with torch.no_grad():
            out, _ = self(img, music, prompt, batch["label"])
            val_loss, _ = compute_cross_entropy(out.logits, music, out.mask)
            self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = Adam(self.projector.parameters(), lr=self.cfg.lr)
        return (
            [optimizer],
            [],
        )


if __name__ == "__main__":
    L.seed_everything(SEED, workers=True)
    cfg = ModelConfig(
        10, concepts=["8bit", "metal"], batch_size=120, model_name="small", lr=1e-1
    )
    music_model = MusicGen.get_pretrained(f"facebook/musicgen-{cfg.model_name}")
    music_model.set_generation_params(use_sampling=True, top_k=250, duration=5)
    concepts_db = TextConcepts.from_musicgen(
        music_model, TokensProvider(cfg.tokens_num), cfg.concepts
    )
    ti_model = TIMusicGen(music_model, concepts_db, cfg)
    projector = ClipProjector(cfg.tokens_num, len(cfg.concepts))
    final_model = ClipTextualInversion(projector, ti_model, cfg)
    dm = ConceptDataModule(concepts_db, cfg.batch_size)
    trainer = L.Trainer(
        callbacks=[],
        enable_checkpointing=False,
        logger=WandbLogger(project="clip-textual-inversion", save_dir=LOGS_PATH),
        log_every_n_steps=10,
        max_epochs=100,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
    )
    trainer.fit(final_model, dm)
