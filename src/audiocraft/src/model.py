from util_tools import compute_cross_entropy, compute_ortho_loss

from torch.utils.data import DataLoader, default_collate
import tqdm
import pytorch_lightning as L
from datasets import load_dataset
from tools.project import INPUT_PATH, LOGS_PATH, OUTPUT_PATH, MODELS_PATH
import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import contextlib
import io
import os
from data import TokensProvider, ConceptDataModule, get_ds, TokensProvider

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_read, audio_write
from audioldm_eval.metrics.fad import FrechetAudioDistance

EXAMPLES_LEN = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

class TransformerTextualInversion(L.LightningModule):
    def __init__(self, text_model, tokenizer, music_model, music_model_conditioner, token_ids, 
                 grad_amplify: float=10.0,
                 entropy_alpha: float=1e1,
                 ortho_alpha: float=1e-2
                 ):
        super().__init__()
        # self.save_hyperparameters()  # Saves all init arguments to the checkpoint
        self.grad_amplify = grad_amplify
        self.entropy_alpha = entropy_alpha
        self.ortho_alpha = ortho_alpha

        self.text_model = text_model
        # self.tokenizer = tokenizer
        self.music_model = music_model
        self.token_ids = token_ids
        self.music_model_conditioner = music_model_conditioner

        
    def _init_text_model(self):
        with torch.no_grad():
            for new_token_id in self.token_ids:
                self.text_model.shared.weight[new_token_id] = self.text_model.shared.weight.mean(dim=0)
        def zero_existing_emb(grad):
            mask = torch.zeros_like(grad)
            for new_token_id in self.token_ids:
                mask[new_token_id] = self.grad_amplify
            return grad * mask

        self.text_model.shared.weight.register_hook(zero_existing_emb)
        
    def on_train_start(self):
        self._init_text_model()

    def forward(self, encoded_music, prompts):
        tokenized_prompt = tokenizer(prompts, return_tensors='pt', padding=True, add_special_tokens=False)
        mask = tokenized_prompt['attention_mask']
        with self.music_model_conditioner.autocast and torch.set_grad_enabled(True):
            x_e = self.text_model(**tokenized_prompt).last_hidden_state
        x_e = self.music_model_conditioner.output_proj(x_e.to(self.music_model_conditioner.output_proj.weight))
        x_e = (x_e * mask.unsqueeze(-1))
        with self.music_model.autocast:
            x = self.music_model.lm.compute_predictions(encoded_music, [], {'description': (x_e, mask)})
        return x
    
    def on_before_optimizer_step(self, optimizer):
        grad_norm = text_model.shared.weight[self.token_ids].norm().item()
        self.log('grad_norm', grad_norm, on_step=True, prog_bar=True)

    def training_step(self, batch, batch_idx):
        music, prompt = batch['encoded_music'], batch['prompt']
        out = self(music, prompt)
        ce_loss, _ = compute_cross_entropy(out.logits, music, out.mask)
        ortho_loss = compute_ortho_loss(self.text_model.shared.weight[batch['batch_tokens']])
        loss = self.entropy_alpha * ce_loss + self.ortho_alpha * ortho_loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("ortho_loss", ortho_loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        music, prompt = batch['encoded_music'], batch['prompt']
        out = self(music, prompt)
        val_loss, _ = compute_cross_entropy(out.logits, music, out.mask)
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        # Optimizer and learning rate scheduler setup
        optimizer = Adam([self.text_model.shared.weight], lr=1e-1)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return ([optimizer], 
                []
                )
class GenEvalCallback(L.Callback):
    def __init__(self, generation_concepts, fad, n_epochs=10):
        super().__init__()
        self.n_epochs = n_epochs
        self.concepts = generation_concepts
        self.fad = fad

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch+1) % self.n_epochs == 0:
            print(f"Generation time at epoch {trainer.current_epoch + 1}")
            for concept in self.concepts:
                response = pl_module.music_model.generate([f'In the style of {TokensProvider(5).get_str(concept)}']*3)
                for a_idx in range(response.shape[0]):
                    music = response[a_idx].cpu()
                    music = music/np.max(np.abs(music.numpy()))
                    path = OUTPUT_PATH("textual-inversion-v3", concept, 'temp', f'music_p{a_idx}')
                    audio_write(path, music, pl_module.music_model.cfg.sample_rate)
                    pl_module.logger.experiment.add_audio(f"{concept} {a_idx}", music, trainer.global_step, sample_rate=pl_module.music_model.cfg.sample_rate)
                with contextlib.redirect_stdout(io.StringIO()):
                    fd_score = self.fad.score(INPUT_PATH('textual-inversion-v3', 'data', 'valid', f'{concept}', 'audio'), OUTPUT_PATH("textual-inversion-v3", concept, 'temp'))
                    os.remove(OUTPUT_PATH("textual-inversion-v3", concept, 'temp_fad_feature_cache.npy'))
                    pl_module.log(f'FAD {concept}', list(fd_score.values())[0])

def get_new_concepts():
    ds = get_ds()
    new_concepts = set()
    def collect_unique(batch):
        new_concepts.update(batch['concept'])
    ds.map(collect_unique, batched=True, batch_size=1000)
    return new_concepts

def append_new_tokens(tokenizer, tokens_by_concept):
    tokens_ids = {}
    idxs = []
    for concept, tokens in tokens_by_concept.items():
        tokenizer.add_tokens(tokens)
        tokens_ids[concept] = tokenizer.convert_tokens_to_ids(tokens)
        idxs.extend(tokens_ids[concept])
    return tokens_ids, idxs

if __name__ == '__main__':

    concepts_to_learn = get_new_concepts()
    concepts_to_learn = ['cluster_0'] 
    ds = get_ds().filter(lambda x: x['concept'] in concepts_to_learn)

    tokens_provider = TokensProvider(5)
    tokens_by_concept = {concept: list(tokens_provider.get(concept)) for concept in concepts_to_learn}

    music_model = MusicGen.get_pretrained('facebook/musicgen-small')
    music_model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=EXAMPLES_LEN
    )
    text_conditioner=list(music_model.lm.condition_provider.conditioners.values())[0]
    tokenizer=text_conditioner.t5_tokenizer
    text_model=text_conditioner.t5

    tokens_ids_by_concept, tokens_ids = append_new_tokens(tokenizer, tokens_by_concept)
    text_model.resize_token_embeddings(len(tokenizer))

    fad = FrechetAudioDistance()
    dm = ConceptDataModule(ds, tokens_provider, tokens_ids_by_concept, music_len=255)
    model = TransformerTextualInversion(text_model, tokenizer, music_model, text_conditioner, tokens_ids)
    tb_logger = L.loggers.TensorBoardLogger(LOGS_PATH, name='textual-inversion-v3')
    trainer = L.Trainer(accelerator='cpu', callbacks=[GenEvalCallback(['cluster_0'], fad)], enable_checkpointing=False, logger=tb_logger, log_every_n_steps=10)
    trainer.fit(model, dm)