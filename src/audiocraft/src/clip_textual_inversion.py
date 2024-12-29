import random


from data_const import train_desc, val_desc
import pytorch_lightning as L
import os


from audiocraft.models import MusicGen
import torch
from tools.project import INPUT_PATH
from data import TextConcepts, TokensProvider, Concept
from losses import compute_cross_entropy
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


class ConceptTensorDataset(Dataset):
    def __init__(self, split: str, concept: Concept):
        ds = torch.load(
            INPUT_PATH("textual-inversion", concept.name, "encoded.pt"),
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
        return len(self.img_ds)

    def __getitem__(self, idx):
        img, label = self.img_ds[idx]
        m_idx = random.randint(0, len(self.concept_ds) - 1)
        m_data = self.concept_ds[m_idx]
        return {
            "img": img,
            "encoded_music": m_data["encoded_music"],
            "prompt": m_data["prompt"],
            "label": label,
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


class ClipProjector(torch.nn.Module):
    def __init__(self, tokens_num):
        super(ClipProjector, self).__init__()
        self.tokens_num = tokens_num
        self.linear = torch.nn.Linear(512, self.tokens_num * 768)

    def forward(self, x):
        x = self.linear(x)
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
        self.music_model.enable_grad()
        self._init_text_model()

    def forward(self, img, music, prompt):
        tokenized = self.music_model.tokenizer(
            prompt, return_tensors="pt", padding=True, add_special_tokens=False
        )
        tokenized = {k: v.to(DEVICE) for k, v in tokenized.items()}

        mask = tokenized["attention_mask"]
        text_with_clip = self.music_model.text_model.shared.weight[
            tokenized["input_ids"]
        ]
        # HARDOCODED AS PROMPT IS HARDOCODED
        text_with_clip[:, -self.cfg.tokens_num :, :] = self.projector(img)
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
            return self.music_model.model.lm.compute_predictions(
                music, [], {"description": (text_emb, mask)}
            )

    def training_step(self, batch, batch_idx):
        self.music_model.enable_grad()

        img, music, prompt = batch["img"], batch["encoded_music"], batch["prompt"]
        out = self(img, music, prompt)
        loss, _ = compute_cross_entropy(out.logits, music, out.mask)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img, music, prompt = batch["img"], batch["encoded_music"], batch["prompt"]
        with torch.no_grad():
            out = self(img, music, prompt)
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
    cfg = ModelConfig(5, concepts=["8bit", "metal"])
    music_model = MusicGen.get_pretrained(f"facebook/musicgen-{cfg.model_name}")
    music_model.set_generation_params(use_sampling=True, top_k=250, duration=5)
    concepts_db = TextConcepts.from_musicgen(
        music_model, TokensProvider(cfg.tokens_num), cfg.concepts
    )
    ti_model = TIMusicGen(music_model, concepts_db, cfg)
    projector = ClipProjector(cfg.tokens_num)
    final_model = ClipTextualInversion(projector, ti_model, cfg)
    dm = ConceptDataModule(concepts_db, cfg.batch_size)
    trainer = L.Trainer(
        callbacks=[],
        enable_checkpointing=False,
        # logger=wandb_logger,
        log_every_n_steps=10,
        max_epochs=1,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
    )
    trainer.fit(final_model, dm)
