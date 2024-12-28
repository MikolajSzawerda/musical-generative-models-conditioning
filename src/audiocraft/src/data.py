from tools.project import INPUT_PATH, LOGS_PATH, OUTPUT_PATH, MODELS_PATH, RAW_PATH
import torch
import os
from datasets import Audio, load_dataset
from datasets import load_dataset
from random import choice
import tqdm
from torch.utils.data import DataLoader, default_collate
from torch.utils.data import TensorDataset, random_split, DataLoader
import pytorch_lightning as L
from copy import deepcopy

DATASET = "concepts-dataset"

train_desc = [
    "the sound of %s",
    "pure %s audio",
    "the recorded %s sound",
    "%s audio sample",
    "recording of %s",
    "high fidelity %s audio",
    "%s sound clip",
    "audio of %s",
    "captured %s sound",
    "%s audio recording",
    "%s recording capture",
    "audio file of %s",
    "isolated %s sound",
    "distinct %s recording",
    "quality %s audio file",
    "high-definition %s sound",
    "the sound recording of %s",
    "audio segment of %s",
    "raw %s audio",
    "%s sound snippet",
    "%s audio track",
    "%s sound fragment",
    "audio recording for %s",
    "sound capture of %s",
    "%s audio file sample",
    "the isolated %s recording",
    "%s recorded audio",
    "pure capture of %s",
    "audio segment capture of %s",
    "the sample of %s audio",
    "the sound file of %s",
    "full recording of %s",
    "%s audio archive",
    "%s sound collection",
    "captured audio of %s",
    "%s isolated sound file",
    "the audio snippet of %s",
    "clean audio of %s",
    "%s audio capture",
    "%s sound extract"
]

val_desc = [
    "audio capture of %s",
    "%s sound recording",
    "pristine %s audio",
    "clear %s recording",
    "the audio of %s",
    "%s audio sample capture",
    "the recorded sound of %s",
    "sample of %s audio",
    "%s audio segment",
    "recorded audio of %s",
    "%s audio",
    "distinct sound of %s",
    "unprocessed %s audio",
    "%s recording",
    "high clarity %s sound",
    "%s recording sample",
    "audio portion of %s",
    "sampled audio of %s",
    "unfiltered %s audio",
    "audio segment for %s",
    "clip of %s audio",
    "the audio snippet for %s",
    "audio portion of %s",
    "%s recorded segment",
    "sampled sound of %s",
    "%s captured in audio",
    "audio excerpt of %s",
    "full audio capture of %s",
    "%s sound archive",
    "audio track of %s",
    "%s in sound format",
    "%s sound recording sample",
    "captured file of %s sound",
    "the distinct sound of %s",
    "high quality %s sound sample",
    "%s in captured audio",
    "pure audio of %s",
    "clean capture of %s audio",
    "recorded file of %s",
    "audio format of %s"
]

NUM_WORKERS = int(os.cpu_count() *0.75)

def get_ds():
    return load_dataset('json', data_files={
                'valid': INPUT_PATH(DATASET, 'metadata_val.json'),
                'train': INPUT_PATH(DATASET, 'metadata_train.json')
                })
def get_hg_ds():
    return load_dataset('mszawerd/concept-dataset')

class TokensProvider:
    def __init__(self, num: int):
        self.num = num
    
    def get(self, base: str):
        return [f'<{base}_{x}>' for x in range(self.num)]
    
    def get_str(self, base: str):
        return ' '.join(self.get(base))

class PromptProvider:
    def __init__(self, prompts_template):
        self.template = prompts_template
    
    def get(self, *args):
        return choice(self.template) % args

class ConceptDataset(torch.utils.data.Dataset):
    def __init__(self, ds, new_tokens_ids, split: str, tokens_provider, sr: int=32000, music_len: int=100, pad_value: float=0.0, base_dir=DATASET):
        self.ds = ds
        if self.ds.cache_files:
            self.base_dir = os.path.dirname(self.ds.cache_files[0]["filename"])
        else:
            raise ValueError("No cache files found in the dataset")
        self.base_dir = INPUT_PATH(base_dir)

        # if split == 'valid':
        #     def map_path(x):
        #         x['audio'] = os.path.join(self.base_dir, x['audio_path'])
        #         return x
        #     self.ds = self.ds.map(map_path).cast_column('audio', Audio(sampling_rate=sr))

        self.encoded = {}
        self.prompter = PromptProvider(val_desc if split == 'valid' else train_desc)
        self.tokens_provider = tokens_provider
        self.music_len = music_len
        self.split = split
        self.concpets = None
        self.tokenized_prompts = {}
        self.tokens_ids = new_tokens_ids
        self._preload_encoded(ds, pad_value)

        
    def _preload_encoded(self, ds, pad_value: float):
        enc_cache = []
        # max_len = -1
        for row in ds:
            path = row['encoded_path']
            encoded = torch.load(os.path.join(self.base_dir, path)).squeeze()
            enc_cache.append(self._random_slice(encoded))
            # max_len = max(max_len, encoded.shape[-1])
        # padded_list = []
        # for enc in enc_cache:
        #     length = enc.shape[-1]
        #     if length < max_len:
        #         shape = list(enc.shape)
        #         shape[-1] = max_len
        #         padded = enc.new_full(shape, pad_value)
        #         padded[..., :length] = enc
        #         padded_list.append(padded)
        #     else:
        #         padded_list.append(enc)

        self.encoded_musics = torch.stack(enc_cache, dim=0).contiguous()
    
    def __len__(self):
        return len(self.ds)
    
    def _random_slice(self, tensor):
        n, k = tensor.shape
        
        if self.music_len <= k:
            # start_col = torch.randint(0, k - self.music_len + 1, (1,)).item()
            start_col = 0
            return tensor[:, start_col:start_col + self.music_len].detach()
        else:
            padding = torch.zeros((n, self.music_len - k), device=tensor.device, dtype=torch.int64)
            return torch.cat((tensor, padding), dim=1).detach()

    def __getitem__(self, idx):
        true_idx = idx
        row = self.ds[true_idx]
        concept = row['concept']
        return {
            'encoded_music': self._random_slice(self.encoded_musics[idx]),
            'concept': concept,
            'prompt': self.prompter.get(self.tokens_provider.get_str(concept)),
            'new_tokens_ids': self.tokens_ids[concept]
            # **({} if self.split == 'train' else 
            #     {
            #         'audio': row['audio']['array']
            #     })
        }
    
    def _get_concepts(self):
        unique_values = set()
        def collect_unique(batch):
            unique_values.update([x.replace("\\", "").split('/')[2] for x in batch['audio_path']])
        self.ds.map(collect_unique, batched=True, batch_size=1000)
        return unique_values
    
    def get_concepts(self):
        if self.concpets is None:
            self.concpets = self._get_concepts()
        return self.concpets
    
    def get_new_tokens(self) -> set[str]:
        res = set()
        for concept in self.get_concepts():
            res.update(self.tokens_provider.get(concept))
        return res
    
    def get_new_tokens_ids(self) -> set[int]:
        return self.tokenizer.convert_tokens_to_ids(self.get_new_tokens())

def collate_fn(batch):
    collated_batch = default_collate(batch)
    collated_batch['batch_tokens'] = torch.unique(torch.cat(collated_batch['new_tokens_ids']))
    return collated_batch

class ConceptTensorDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, tokens_provider, tokens_ids, sr: int=32000):
        ds=torch.load(INPUT_PATH('encoded.pt'))[:225, :, :].cpu()
        ds = TensorDataset(ds)
        train_ds, val_ds = random_split(ds, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
        self.split = split
        if self.split == 'train':
            self.ds = train_ds
        else:
            self.ds = val_ds
        self.prompter = PromptProvider(val_desc if split == 'valid' else train_desc)
        self.tokens_provider = tokens_provider
        self.tokens_ids = tokens_ids
    
    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        true_idx = idx
        row = self.ds[true_idx]
        concept = '8bit'
        return {
            'encoded_music': row[0],
            'concept': concept,
            'prompt': self.prompter.get(self.tokens_provider.get_str(concept)),
            'new_tokens_ids': self.tokens_ids[concept]
            # **({} if self.split == 'train' else 
            #     {
            #         'audio': row['audio']['array']
            #     })
        }

class ConceptDataModule(L.LightningDataModule):
    def __init__(self, ds, tokens_provider, tokens_ids, music_len: int = 255, batch_size: int = 5):
        super().__init__()
        self.tokens_provider = tokens_provider
        self.tokens_ids = tokens_ids
        self.music_len = music_len
        self.batch_size = batch_size
        self.ds = ds
   
    def setup(self, stage: str):
        self.train_ds = ConceptDataset(self.ds['train'], self.tokens_ids,'train', self.tokens_provider, music_len=self.music_len)
        self.val_ds = ConceptDataset(self.ds['valid'], self.tokens_ids, 'valid', self.tokens_provider, music_len=self.music_len)
    
    # def setup(self, stage: str):
    #     self.train_ds = ConceptTensorDataset('train', self.tokens_provider, self.tokens_ids)
    #     self.val_ds = ConceptTensorDataset('valid', self.tokens_provider, self.tokens_ids)

    def get_new_tokens(self)->list[str]:
        new_tokens = self.train_ds.get_new_tokens()
        new_tokens.update(self.val_ds.get_new_tokens())
        return list(new_tokens)
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, collate_fn=collate_fn, num_workers=NUM_WORKERS, persistent_workers=True, shuffle=True)
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size, collate_fn=collate_fn, num_workers=NUM_WORKERS, persistent_workers=True)

class EvalDataModule(L.LightningDataModule):
    def __init__(self, ds, tokens_provider, tokens_ids, music_len: int = 255, batch_size: int = 5, base_dir=DATASET):
        super().__init__()
        self.tokens_provider = tokens_provider
        self.tokens_ids = tokens_ids
        self.music_len = music_len
        self.batch_size = batch_size
        self.ds = ds
        self.base_dir = base_dir
   
    def setup(self, stage: str):
        self.train_ds = ConceptDataset(self.ds['train'], self.tokens_ids,'train', self.tokens_provider, music_len=self.music_len, base_dir=self.base_dir)
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, collate_fn=collate_fn, num_workers=NUM_WORKERS, persistent_workers=True, shuffle=True)
    
    def val_dataloader(self) -> DataLoader:
        return []


if __name__ == '__main__':
    ds = get_ds()['train']
    cds = ConceptDataset(ds, None, 'train', None)
    # print(next(iter(ConceptTensorDataset('train', TokensProvider(5), {'8bit': [1, 43, 2]}))))