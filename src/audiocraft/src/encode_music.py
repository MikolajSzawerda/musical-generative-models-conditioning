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
DATASET = "eval-ds"

# parser = argparse.ArgumentParser()
# parser.add_argument('-c', '--concept', required=True)
# parser.add_argument('-s', '--split', required=True)
# args=parser.parse_args()

def remove_current(concept, split):
    name = 'metadata_train.json' if split == 'train' else 'metadata_val.json'
    with open(INPUT_PATH(DATASET, name), 'r') as fh:
        data = json.load(fh)
    filtered = [x for x in data if x.get('concept') != concept]
    with open(INPUT_PATH(DATASET, name), 'w') as fh:
        json.dump(filtered, fh, indent=4, ensure_ascii=False)


def extend_current(data, split):
    name = 'metadata_train.json' if split == 'train' else 'metadata_val.json'
    with open(INPUT_PATH(DATASET, name), 'r') as fh:
        fdata = json.load(fh)
    fdata.extend(data)
    with open(INPUT_PATH(DATASET, name), 'w') as fh:
        json.dump(fdata, fh, indent=4, ensure_ascii=False)

def clean_directory(directory):
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)

def encode(concept: str, split: str):
    remove_current(concept, split)
    # concept, split= args.concept, args.split
    base_path = INPUT_PATH(DATASET, 'data', split, concept)
    clean_directory(Path(INPUT_PATH(DATASET, "data", split, concept, 'encoded')))
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

import os
import random
import shutil

def copy_random_audio_files(parent_directory, num_files=20):
    """
    For each subfolder in 'parent_directory':
      - Create a 'fad' subdirectory (if not already created).
      - Randomly copy 'num_files' files from the 'audio' subdirectory into 'fad'.
    """
    for item in os.listdir(parent_directory):
        subfolder_path = os.path.join(parent_directory, item)

        if not os.path.isdir(subfolder_path):
            continue
        
        fad_dir = os.path.join(subfolder_path, 'fad')
        os.makedirs(fad_dir, exist_ok=True)
        audio_dir = os.path.join(subfolder_path, 'audio')

        if os.path.isdir(audio_dir):
            all_files = [
                f for f in os.listdir(audio_dir)
                if os.path.isfile(os.path.join(audio_dir, f))
            ]

            num_to_copy = min(num_files, len(all_files))
            selected_files = random.sample(all_files, num_to_copy)
            for audio_file in selected_files:
                src = os.path.join(audio_dir, audio_file)
                dst = os.path.join(fad_dir, audio_file)
                shutil.copy2(src, dst)

            print(f"Copied {num_to_copy} files from '{audio_dir}' to '{fad_dir}'")
        else:
            print(f"No 'audio' directory found in '{subfolder_path}'")

if __name__ == '__main__':
    # copy_random_audio_files(INPUT_PATH(DATASET, "data", "valid"))

    concepts = ['bells']
    # concepts = ['two-steps', 'chillout', 'metal-solos', '8bit-slow', 'choir', '8bit', 'piano', '80s-synth', 'saxophone-chillout', 'b-minor-rock', 'bells', 'pirates']
    for split in ['train']:
        for concept in concepts:
            encode(concept, split)
