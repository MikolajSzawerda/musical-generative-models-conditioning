from tools.project import INPUT_PATH, LOGS_PATH, OUTPUT_PATH, MODELS_PATH, RAW_PATH

import audiocraft
from audiocraft.models import MusicGen
from audiocraft.utils.notebook import display_audio
import torch
from gradio.cli.commands.components.publish import colors
from omegaconf import DictConfig
from torch import set_grad_enabled
from torch.onnx.symbolic_opset9 import tensor
from torchviz import make_dot
import typing as tp
from audiocraft.modules.conditioners import ConditioningAttributes
import tqdm
import torch
from audiocraft.data.audio import audio_read, audio_write
from audiocraft.data.audio_utils import convert_audio_channels, convert_audio
import umap
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from torch.utils.tensorboard import SummaryWriter
from sklearn.decomposition import PCA
from random import shuffle
from torch.utils.data import TensorDataset, random_split, DataLoader
from audioldm_eval.metrics.fad import FrechetAudioDistance
import os
import contextlib
import io
from types import SimpleNamespace
import json
import datasets
from datasets import Dataset
import pandas as pd
import shutil
from datasets import load_dataset, Audio

DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
MAX_NUM = 20

def load_music_gen():
    model = MusicGen.get_pretrained('facebook/musicgen-small')
    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=5
    )
    return model
encodec = load_music_gen().compression_model

def clear_if_exists(dir_name):
    if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
    os.makedirs(dir_name, exist_ok=True)

def gen_genre_ds(music_ds, label, step, audio_for_train=False):
    new_ds = []
    if step == 'train':
        clear_if_exists(INPUT_PATH('textual-inversion-v2', 'data', step, label, 'encoded'))
    if step == 'valid' or audio_for_train:
        clear_if_exists(INPUT_PATH('textual-inversion-v2', 'data', step, label, 'audio'))
        clear_if_exists(INPUT_PATH('textual-inversion-v2', 'data', step, label, 'encoded'))
    for idx, row in tqdm.tqdm(enumerate(music_ds)):
        path = row['path']['path']
        audio_path = None
        if step == 'valid' or audio_for_train:
            audio_path = os.path.join('data', step, label, 'audio', os.path.basename(path))
            dest = INPUT_PATH('textual-inversion-v2', audio_path)
            shutil.copy2(path, dest)
        enc_path = os.path.join('data', step, label, 'encoded', f'music_{idx}.pt')
        dest = INPUT_PATH('textual-inversion-v2', enc_path)
        with torch.no_grad():
            inp = torch.from_numpy(row['path']['array'])[None][None]
            inp = inp.type(next(iter(encodec.named_parameters()))[1].dtype)
            encoded_music, _ = encodec.encode(inp.to(DEVICE))
            torch.save(encoded_music.cpu(), dest)
        new_ds.append({
            'audio_path': audio_path,
            'encoded_path': enc_path,
            'tag': row['tag_top188'],
            'track_id': row['track_id'],
            'artist_name': row['artist_name']
        })
    return new_ds




tag_filter = lambda labels, x: all(g in x['tag_top188'] for g in labels)
attr_filter = lambda labels, x: all(g in x['pseudo_attribute'] for g in labels)
data = [
    ('jazz',lambda x: tag_filter(['jazz'], x)),
    ('melancholy',lambda x: attr_filter(['melancholy'], x)),
    ('calm',lambda x: attr_filter(['calm'], x)),
    ('dramatic',lambda x: attr_filter(['dramatic'], x)),
    ('guitar_solo',lambda x: attr_filter(['guitar solo'], x)),
    ('drum_fills',lambda x: attr_filter(['drum fills'], x)),
    ('upbeat',lambda x: attr_filter(['upbeat'], x)),
    ('ambient',lambda x: tag_filter(['ambient'], x)),
]

if __name__ == '__main__':
    def shared(ex):
        ex['path']=INPUT_PATH('musictag', ex['path'])
        return ex
    ds = load_dataset("seungheondoh/LP-MusicCaps-MTT").map(shared)
    
    def make_data(ds, label, filter_func):
        def ds_fetch(step):
            filtered_ds = ds[step].filter(filter_func)
            return filtered_ds.take(min(MAX_NUM, len(filtered_ds))).cast_column('path', Audio(sampling_rate=32000))
        return (
            gen_genre_ds(ds_fetch('valid'), label, 'valid'), 
            gen_genre_ds(ds_fetch('train'), label, 'train', audio_for_train=True)
        )
    def save_json(ds: list, name):
        pd.DataFrame(ds).to_json(INPUT_PATH('textual-inversion-v2', name), force_ascii=False, orient='records')

    res = list(zip(*[make_data(ds, label, filter_func) for label, filter_func in data]))
    save_json(res[0][0], 'metadata_val.json')
    save_json(res[1][0], 'metadata_train.json')