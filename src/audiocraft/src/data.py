from tools.project import INPUT_PATH, LOGS_PATH, OUTPUT_PATH, MODELS_PATH, RAW_PATH
import torch
import os
from datasets import Audio, load_dataset
from datasets import load_dataset
from random import choice
import tqdm

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

def get_ds():
    return load_dataset('json', data_files={
                'valid': INPUT_PATH('textual-inversion-v2', 'metadata_val.json'),
                'train': INPUT_PATH('textual-inversion-v2', 'metadata_train.json')
                })

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
    def __init__(self, ds, tokenizer, split: str, sr: int=32000, tokens_num: int=1, music_len: int=100):
        self.ds = ds
        self.tokenizer = tokenizer

        if self.ds.cache_files:
            self.base_dir = os.path.dirname(self.ds.cache_files[0]["filename"])
        else:
            raise ValueError("No cache files found in the dataset")
        self.base_dir = INPUT_PATH('textual-inversion-v2')

        if split == 'valid':
            def map_path(x):
                x['audio_path'] = os.path.join(self.base_dir, x['audio_path'])
                x['audio'] = os.path.join(self.base_dir, x['audio_path'])
                return x
            self.ds = self.ds.map(map_path).cast_column('audio', Audio(sampling_rate=sr))

        self.encoded = {}
        self.tokens_num = tokens_num
        self.prompter = PromptProvider(val_desc if split == 'valid' else train_desc)
        self.tokens_provider = TokensProvider(tokens_num)
        self.music_len = music_len
        self.split = split
        self.concpets = None
        self.tokenized_prompts = {}
        self.tokens_ids = {}
    
    def __len__(self):
        return len(self.ds)
    
    def _random_slice(self, tensor):
        n, k = tensor.shape
        
        if self.music_len <= k:
            start_col = torch.randint(0, k - self.music_len + 1, (1,)).item()
            return tensor[:, start_col:start_col + self.music_len]
        else:
            padding = torch.zeros((n, self.music_len - k), device=tensor.device)
            return torch.cat((tensor, padding), dim=1)
    
    def __getitem__(self, idx):
        row = self.ds[idx]
        path = row['encoded_path']
        if path not in self.encoded:
            self.encoded[path] = torch.load(os.path.join(self.base_dir, path)).squeeze()
        y = path.replace("\\", "").split('/')[2]
        if y not in self.tokens_ids:
            self.tokens_ids[y] = self.tokenizer.convert_tokens_to_ids(self.tokens_provider.get(y))
        prompt = self.prompter.get(self.tokens_provider.get_str(y))
        # if prompt not in self.tokenized_prompts:
        #     self.tokenized_prompts[prompt] = self.tokenizer([prompt], return_tensors='pt', padding=True, add_special_tokens=False)
        return {
            'encoded_music': self._random_slice(self.encoded[path]),
            'prompt': prompt,
            'new_token_ids': self.tokens_ids[y],
            **({} if self.split == 'train' else 
                {
                    'audio': row['audio_path']['array']
                })
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

if __name__ == '__main__':
    dl = torch.utils.data.DataLoader(ConceptDataset('valid'), batch_size=2)
    for batch in tqdm.tqdm(dl):
        print(batch)