from data import TextConcepts, Concept
from losses import compute_cross_entropy, compute_ortho_loss
import pytorch_lightning as L
from tools.project import MODELS_PATH
import torch
import torch.nn.functional as F
from torch.optim import Adam
from data import ConceptDataModule, get_ds, TokensProvider

from audiocraft.models import MusicGen
import dataclasses
import logging
from toolz import concat
from callbacks import ConceptEmbeds
from data_const import Datasets

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


@dataclasses.dataclass
class ModelConfig:
    tokens_num: int
    grad_amplify: float = 10.0
    entropy_alpha: float = 1e1
    ortho_alpha: float = 1e-2
    lr: float = 1e-1
    model_name: str = "small"
    examples_len: int = 5
    concepts: list[str] = tuple()
    batch_size: int = 10


class TIMusicGen:
    def __init__(self, model: MusicGen, concepts_db: TextConcepts, cfg: ModelConfig):
        self.model = model
        self.text_conditioner = list(model.lm.condition_provider.conditioners.values())[
            0
        ]
        self.tokenizer = self.text_conditioner.t5_tokenizer
        self.text_model = self.text_conditioner.t5
        self.text_weights = self.text_model.shared.weight
        self.db = concepts_db
        self.cfg = cfg

    @torch.no_grad()
    def init_model_random(self, sigma=0.1):
        weight_mean = self.text_model.shared.weight.mean(dim=0)
        ids = self.db.all_token_ids
        assert len(self.text_weights) >= max(ids)
        noise = (
            torch.randn(
                (len(ids), weight_mean.shape[0]),
                device=weight_mean.device,
                dtype=weight_mean.dtype,
            )
            * sigma
        )
        self.text_weights[ids] = weight_mean.expand(len(ids), -1) + noise

    @torch.no_grad()
    def init_model(self, embedings: dict[str, ConceptEmbeds]):
        def init_concept(concept: Concept):
            data = embedings[concept.name]
            logger.info(f"Loaded embeds for {concept.name} from {data.epoch} epoch")
            ids = concept.token_ids
            for i, idx in enumerate(ids):
                self.text_weights[idx] = data.embeds[i].detach()

        self.db.execute(init_concept)

    def enable_grad(self, val=True):
        self.model.lm.requires_grad_(val)
        self.text_model.requires_grad_(val)
        self.text_conditioner.finetune = val


class TransformerTextualInversion(L.LightningModule):
    def __init__(
        self,
        model: TIMusicGen,
        cfg: ModelConfig,
    ):
        super().__init__()
        self.cfg = cfg
        self.model = model

    @staticmethod
    def get_ti_model(model: MusicGen, cfg: ModelConfig) -> TIMusicGen:
        concepts_db = TextConcepts.from_musicgen(
            model, TokensProvider(cfg.tokens_num), cfg.concepts
        )
        return TIMusicGen(model, concepts_db, cfg)

    @classmethod
    def from_previous_run(
        cls,
        base_dir: str,
        run_name: str,
        music_model: MusicGen,
        cfg: ModelConfig,
    ):
        ti_model = TransformerTextualInversion.get_ti_model(music_model, cfg)
        raw_embeds = torch.load(MODELS_PATH(base_dir, f"{run_name}-best.pt"))
        embeds = {
            c: ConceptEmbeds(data["epoch"], data["embeds"])
            for c, data in raw_embeds.items()
        }
        ti_model.init_model(embeds)
        return cls(ti_model, cfg)

    @classmethod
    def from_musicgen(
        cls,
        music_model: MusicGen,
        cfg: ModelConfig,
    ):
        ti_model = TransformerTextualInversion.get_ti_model(music_model, cfg)
        ti_model.init_model_random()
        return cls(ti_model, cfg)

    def _init_text_model(self):
        def zero_existing_emb(grad):
            mask = torch.zeros_like(grad)
            for new_token_id in self.model.db.all_token_ids:
                mask[new_token_id] = self.cfg.grad_amplify
            return grad * mask

        self.model.text_weights.register_hook(zero_existing_emb)

    def on_train_start(self):
        self.model.enable_grad()
        self._init_text_model()

    def forward(self, encoded_music, prompts):
        tokenized_prompt = self.model.tokenizer(
            prompts, return_tensors="pt", padding=True, add_special_tokens=False
        )
        tokenized_prompt = {k: v.to(DEVICE) for k, v in tokenized_prompt.items()}
        mask = tokenized_prompt["attention_mask"]
        with self.model.text_conditioner.autocast and torch.set_grad_enabled(True):
            x_e = self.model.text_model(**tokenized_prompt).last_hidden_state
        x_e = self.model.text_conditioner.output_proj(
            x_e.to(self.model.text_conditioner.output_proj.weight)
        )
        x_e = x_e * mask.unsqueeze(-1)
        with self.model.model.autocast:
            x = self.model.model.lm.compute_predictions(
                encoded_music, [], {"description": (x_e, mask)}
            )
        return x

    def on_before_optimizer_step(self, optimizer):
        grad_norm = (
            self.model.text_weights.grad[self.model.db.all_token_ids].norm().item()
        )
        self.log("grad_norm", grad_norm, on_epoch=True, prog_bar=True)

    def _compute_cr(self, concepts: list[str], margin: float = 1.5):
        nr, unique_concepts, res = (
            lambda x: x / (torch.norm(x) + 1e-8),
            set(concepts),
            0.0,
        )
        if len(unique_concepts) < 2:
            return res
        token_ids = list(
            concat([self.model.db.concepts[c].token_ids for c in unique_concepts])
        )
        concepts_embedings = self.model.text_weights[token_ids]
        p_c = 0
        for i in range(len(unique_concepts)):
            for j in range(i + 1, len(unique_concepts)):
                dist = torch.norm(
                    nr(concepts_embedings[i]) - nr(concepts_embedings[j]), p="fro"
                )
                res += F.relu(margin - dist)
                p_c += 1
        if p_c > 0:
            return res / p_c
        return 0.0

    def training_step(self, batch, batch_idx):
        self.model.enable_grad()

        music, prompt = batch["encoded_music"], batch["prompt"]
        out = self(music, prompt)
        # ce_loss = self._compute_ce_by_concept(batch['concept'], music, out)
        ce_loss, _ = compute_cross_entropy(out.logits, music, out.mask)
        if self.cfg.tokens_num > 1:
            ortho_loss = compute_ortho_loss(
                self.model.text_weights[batch["batch_tokens"]]
            )
            self.log(
                "ortho_loss", ortho_loss, on_step=False, on_epoch=True, prog_bar=True
            )
        else:
            ortho_loss = 0.0
        cr_loss = self._compute_cr(batch["concept"])
        self.log("cr_loss", cr_loss, on_step=False, on_epoch=True, prog_bar=True)

        # loss = self.entropy_alpha * ce_loss + self.ortho_alpha * ortho_loss + self.prev_grad*1e-2
        loss = (
            self.cfg.entropy_alpha * ce_loss
            + self.cfg.ortho_alpha * ortho_loss
            + cr_loss
        )
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        music, prompt = batch["encoded_music"], batch["prompt"]
        with torch.no_grad():
            out = self(music, prompt)
            val_loss, _ = compute_cross_entropy(out.logits, music, out.mask)
            # val_loss = self._compute_ce_by_concept(batch['concept'], music, out)
            self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = Adam([self.model.text_weights], lr=self.cfg.lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return (
            [optimizer],
            [],
            # [{"scheduler": scheduler, "interval": "epoch"}]
        )


def append_new_tokens(tokenizer, tokens_by_concept):
    tokens_ids = {}
    idxs = []
    for concept, tokens in tokens_by_concept.items():
        tokenizer.add_tokens(tokens)
        tokens_ids[concept] = tokenizer.convert_tokens_to_ids(tokens)
        idxs.extend(tokens_ids[concept])
    return tokens_ids, idxs


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    cfg = ModelConfig(10, concepts=["cluster_0"])
    model_name = f"facebook/musicgen-{cfg.model_name}"
    music_model = MusicGen.get_pretrained(model_name)
    music_model.set_generation_params(
        use_sampling=True, top_k=250, duration=cfg.examples_len
    )
    ds = get_ds(Datasets.TEXTUAL_INVERSION_V3).filter(
        lambda x: x["concept"] == "cluster_0"
    )

    model = TransformerTextualInversion.from_musicgen(music_model, cfg)
    cds = ConceptDataModule(ds, model.model.db, Datasets.TEXTUAL_INVERSION_V3)
    cds.setup("train")
    dl = cds.train_dataloader()
    batch = next(iter(dl))
    model.training_step(batch, 0)
