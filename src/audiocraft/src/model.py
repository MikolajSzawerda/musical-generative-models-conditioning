from util_tools import compute_cross_entropy, compute_ortho_loss

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

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_read, audio_write
from audioldm_eval.metrics.fad import FrechetAudioDistance

EXAMPLES_LEN = 5
TOKENS_NUM = 1
BATCH_SIZE = 5
GRAD_AMP = 10.0
ENTROPY_ALPHA = 1e1
ORTHO_ALPHA = 1e-2
LR = 1e-1
MODEL = 'facebook/musicgen-small'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

class TransformerTextualInversion(L.LightningModule):
    def __init__(self, text_model, tokenizer, music_model, music_model_conditioner, token_ids, 
                 tokens_num,
                 grad_amplify: float=10.0,
                 entropy_alpha: float=1e1,
                 ortho_alpha: float=1e-2,
                 lr: float=1e-1
                 ):
        super().__init__()
        # self.save_hyperparameters()  # Saves all init arguments to the checkpoint
        self.grad_amplify = grad_amplify
        self.entropy_alpha = entropy_alpha
        self.ortho_alpha = ortho_alpha

        self.text_model = text_model
        self.tokenizer = tokenizer
        self.music_model = music_model
        self.token_ids = token_ids
        self.music_model_conditioner = music_model_conditioner
        self.lr = lr
        self.prev_grad = 0
        self.tokens_num = tokens_num

        
    def _init_text_model(self):
        with torch.no_grad():
            for new_token_id in self.token_ids:
                self.text_model.shared.weight[new_token_id] = self.text_model.shared.weight.mean(dim=0)
        def zero_existing_emb(grad):
            mask = torch.zeros_like(grad)
            for new_token_id in self.token_ids:
                mask[new_token_id] = self.grad_amplify
            self.prev_grad =  (grad * (1-(mask / self.grad_amplify))).norm().item()
            return grad * mask

        self.text_model.shared.weight.register_hook(zero_existing_emb)
        
    def on_train_start(self):
        self.music_model.lm.requires_grad_(True)
        self.text_model.requires_grad_(True)
        self.music_model_conditioner.finetune=True
        self._init_text_model()

    def forward(self, encoded_music, prompts):
        tokenized_prompt = self.tokenizer(prompts, return_tensors='pt', padding=True, add_special_tokens=False)
        tokenized_prompt = {k: v.to(DEVICE) for k,v in tokenized_prompt.items()}
        mask = tokenized_prompt['attention_mask']
        # print("SHAPE:", encoded_music)
        with self.music_model_conditioner.autocast:
            x_e = self.text_model(**tokenized_prompt).last_hidden_state
        x_e = self.music_model_conditioner.output_proj(x_e.to(self.music_model_conditioner.output_proj.weight))
        x_e = (x_e * mask.unsqueeze(-1))
        with self.music_model.autocast:
            x = self.music_model.lm.compute_predictions(encoded_music, [], {'description': (x_e, mask)})
        return x
    
    def on_before_optimizer_step(self, optimizer):
        self.log('prev_grad', self.prev_grad, on_epoch=True, prog_bar=True)

        grad_norm = self.text_model.shared.weight.grad[self.token_ids].norm().item()
        self.log('grad_norm', grad_norm, on_epoch=True, prog_bar=True)

    def training_step(self, batch, batch_idx):
        self.music_model.lm.requires_grad_(True)
        self.text_model.requires_grad_(True)
        self.music_model_conditioner.finetune=True

        music, prompt = batch['encoded_music'], batch['prompt']
        out = self(music, prompt)
        ce_loss, _ = compute_cross_entropy(out.logits, music, out.mask)
        if self.tokens_num > 1:
            ortho_loss = compute_ortho_loss(self.text_model.shared.weight[batch['batch_tokens']])
            self.log("ortho_loss", ortho_loss, on_step=False, on_epoch=True, prog_bar=True)
        else:
            ortho_loss = 0.0
        loss = self.entropy_alpha * ce_loss + self.ortho_alpha * ortho_loss + self.prev_grad*1e-2
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        music, prompt = batch['encoded_music'], batch['prompt']
        with torch.set_grad_enabled(False):
            out = self(music, prompt)
            val_loss, _ = compute_cross_entropy(out.logits, music, out.mask)
            self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        # Optimizer and learning rate scheduler setup
        optimizer = Adam([self.text_model.shared.weight], lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return ([optimizer], 
                []
                )
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

class GenEvalCallback(L.Callback):
    def __init__(self, generation_concepts, fad, tokens_num, n_epochs=10):
        super().__init__()
        self.n_epochs = n_epochs
        self.concepts = generation_concepts
        self.fad = fad
        self.tokens_num = tokens_num
    
    def _audio_to_spectrogram_image(self, audio, sr):
        if audio.ndim > 1:
            audio = np.squeeze(audio, axis=0)
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=sr/2)
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # Plot the mel-spectrogram
        fig, ax = plt.subplots(figsize=(6,4), dpi=150)
        librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=sr/2, ax=ax, cmap="magma")
        ax.set_title('Mel-Spectrogram')
        ax.set_xlabel("Time")
        ax.set_ylabel("Mel Frequency")
        
        # Convert plot to numpy array
        fig.canvas.draw()
        spectrogram_image = np.array(fig.canvas.renderer.buffer_rgba())
        plt.close(fig)
        return spectrogram_image

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch) % self.n_epochs != 0:
            return
        print(f"Generation time at epoch {trainer.current_epoch + 1}")
        for concept in self.concepts:
            response = pl_module.music_model.generate([f'In the style of {TokensProvider(self.tokens_num).get_str(concept)}']*10)
            audio_list = []
            # table = wandb.Table(columns=["audio", "spectrogram"])
            img_list = []
            for a_idx in range(response.shape[0]):
                music = response[a_idx].cpu()
                music = music/np.max(np.abs(music.numpy()))
                path = OUTPUT_PATH("textual-inversion-v3", concept, 'temp', f'music_p{a_idx}')
                audio_write(path, music, pl_module.music_model.cfg.sample_rate)

                spectrogram = self._audio_to_spectrogram_image(music.numpy(), pl_module.music_model.cfg.sample_rate)
                audio_wdb = wandb.Audio(
                        path+'.wav', 
                        sample_rate=pl_module.music_model.cfg.sample_rate, 
                        caption=f"{concept} audio {a_idx}"
                    )
                spec_wdb = wandb.Image(spectrogram, caption=f"Spectrogram {a_idx}")
                # table.add_data(audio_wdb, spec_wdb)
                audio_list.append(
                    audio_wdb
                )
                # table.add_data(audio_wdb, spec_wdb)
                img_list.append(
                    spec_wdb
                )
                # pl_module.logger.experiment.add_audio(f"{concept} {a_idx}", music, trainer.global_step, sample_rate=pl_module.music_model.cfg.sample_rate)
            pl_module.logger.experiment.log({f"{concept}_audio": audio_list[:5], f"{concept}_spec": img_list[:5],"global_step": trainer.global_step})
            # pl_module.logger.experiment.log({f"{concept} audio_spec": table, "global_step": trainer.global_step})
            fads = []
            with contextlib.redirect_stdout(io.StringIO()):
                fd_score = self.fad.score(INPUT_PATH('textual-inversion-v3', 'data', 'valid', f'{concept}', 'fad'), OUTPUT_PATH("textual-inversion-v3", concept, 'temp'))
                os.remove(OUTPUT_PATH("textual-inversion-v3", concept, 'temp_fad_feature_cache.npy'))
                if isinstance(fd_score, int):
                    return
                val = list(fd_score.values())[0]*1e-5
                pl_module.log(f'FAD {concept}', val)
                fads.append(val)
            pl_module.log(f'fad_avg', np.mean(fads))
            

class SaveEmbeddingsCallback(L.Callback):
    def __init__(self, save_path, concepts, tokens_ids_by_concept, weights, n_epochs=10):
        super().__init__()
        self.save_path = save_path
        self.concepts = concepts
        self.best_score = {c: float("inf") for c in concepts}
        self.best_file_path = None
        self.tokens_ids_by_concept = tokens_ids_by_concept
        self.weights = weights
        self.best_embeds = {
            c: weights[tokens_ids_by_concept[c]].detach().cpu() for c in concepts
        }
        self.n_epochs = n_epochs

    def on_validation_end(self, trainer, pl_module):
        if (trainer.current_epoch) % self.n_epochs != 0:
            return
        for concept in self.concepts:
            metrics = trainer.callback_metrics
            current_score = metrics.get(f'FAD {concept}')

            if current_score is None or current_score > self.best_score[concept]:
                break

            self.best_score[concept] = current_score
            self.best_embeds[concept] = self.weights[self.tokens_ids_by_concept[concept]].detach().cpu()

            wandb_logger = trainer.logger
            if isinstance(wandb_logger, WandbLogger):
                run_name = wandb_logger.experiment.name
            else:
                run_name = str(uuid.uuid4())
            save_file_path = os.path.join(self.save_path, f"{run_name}-best.pt")
            torch.save(self.best_embeds, save_file_path)

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
    concepts_to_learn = ['8bit'] 
    ds = get_ds().filter(lambda x: x['concept'] in concepts_to_learn)

    tokens_provider = TokensProvider(TOKENS_NUM)
    tokens_by_concept = {concept: list(tokens_provider.get(concept)) for concept in concepts_to_learn}

    music_model = MusicGen.get_pretrained(MODEL)
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

    fad = FrechetAudioDistance(verbose=True, use_pca=True, use_activation=True)
    dm = ConceptDataModule(ds, tokens_provider, tokens_ids_by_concept, music_len=255, batch_size=BATCH_SIZE)
    model = TransformerTextualInversion(text_model, tokenizer, music_model, text_conditioner, tokens_ids, grad_amplify=GRAD_AMP, lr=LR, ortho_alpha=ORTHO_ALPHA, entropy_alpha=ENTROPY_ALPHA)
    # tb_logger = L.loggers.TensorBoardLogger(LOGS_PATH, name='textual-inversion-v3')
    wandb_logger = WandbLogger(project='textual-inversion-v3', save_dir=LOGS_PATH)
    wandb_logger.experiment.config['batch_size'] = BATCH_SIZE
    wandb_logger.experiment.config['examples_len'] = EXAMPLES_LEN
    wandb_logger.experiment.config['tokens_num'] = TOKENS_NUM
    wandb_logger.experiment.config['model'] = MODEL
    wandb_logger.experiment.config['concepts'] = concepts_to_learn
    wandb_logger.experiment.config['lr'] = LR
    wandb_logger.experiment.config['ortho_alpha'] = ORTHO_ALPHA
    wandb_logger.experiment.config['entropy_alpha'] = ENTROPY_ALPHA

    quick_save_cl = SaveEmbeddingsCallback(LOGS_PATH('embeds'), concepts_to_learn, tokens_ids_by_concept, text_model.shared.weight)
    trainer = L.Trainer(callbacks=[GenEvalCallback(concepts_to_learn, fad), quick_save_cl], enable_checkpointing=False, logger=wandb_logger, log_every_n_steps=10, profiler='advanced', max_epochs=2)
    trainer.fit(model, dm)