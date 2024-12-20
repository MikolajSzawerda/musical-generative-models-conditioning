import tqdm
from datasets import load_dataset
from tools.project import INPUT_PATH, LOGS_PATH, OUTPUT_PATH, MODELS_PATH
import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
import contextlib
import io
import os
import argparse
import json

from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_read, audio_write
from audiocraft.data.audio_utils import convert_audio_channels, convert_audio
from pathlib import Path
import uuid
import shutil

DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model = MusicGen.get_pretrained('facebook/musicgen-small')
model.set_generation_params(
    use_sampling=True,
    top_k=250,
    duration=5
)

# parser = argparse.ArgumentParser()
# parser.add_argument('-c', '--concept', required=True)
# parser.add_argument('-s', '--split', required=True)
# args=parser.parse_args()

def remove_current(concept, split):
    name = 'metadata_train.json' if split == 'train' else 'metadata_val.json'
    with open(INPUT_PATH('textual-inversion-v3', name), 'r') as fh:
        data = json.load(fh)
    filtered = [x for x in data if x.get('concept') != concept]
    with open(INPUT_PATH('textual-inversion-v3', name), 'w') as fh:
        json.dump(filtered, fh, indent=4, ensure_ascii=False)


def extend_current(data, split):
    name = 'metadata_train.json' if split == 'train' else 'metadata_val.json'
    with open(INPUT_PATH('textual-inversion-v3', name), 'r') as fh:
        fdata = json.load(fh)
    fdata.extend(data)
    with open(INPUT_PATH('textual-inversion-v3', name), 'w') as fh:
        json.dump(fdata, fh, indent=4, ensure_ascii=False)

def clean_directory(directory):
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)

def encode(concept: str, split: str):
    remove_current(concept, split)
    # concept, split= args.concept, args.split
    base_path = INPUT_PATH('textual-inversion-v3', 'data', split, concept)
    clean_directory(Path(INPUT_PATH('textual-inversion-v3', "data", split, concept, 'encoded')))
    audio_files = [f.name for f in Path(os.path.join(base_path, 'audio')).iterdir() if f.is_file()]
    new_rows = []
    for filename in tqdm.tqdm(audio_files):
        path = os.path.join('data', split, concept, 'audio', filename)
        audio_path = os.path.join(base_path, 'audio', filename)
        enc_path =  os.path.join('data', split, concept, 'encoded', filename.replace('.wav', '.pt'))
        dest = os.path.join(base_path, 'encoded', filename.replace('.wav', '.pt'))
        with torch.no_grad():
            music, sr = audio_read(audio_path)
            music = music[None]
            music = convert_audio(music, sr, 32000, 1)
            encoded_music, _ = model.compression_model.encode(music.to(DEVICE))
            torch.save(encoded_music.cpu().type(torch.int64), dest)
        new_rows.append({
            'audio_path': path,
            'encoded_path': enc_path,
            'concept': concept,
            'track_id': str(uuid.uuid4()),
        })
    extend_current(new_rows, split)

if __name__ == '__main__':
    concepts = ['8bit', 'ajfa', 'caravan', 'ichika', 'metal', 'oim', 'synthwave']
    for split in ['valid']:
        for concept in concepts:
            encode(concept, split)
