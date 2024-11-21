import tqdm
from tools.project import INPUT_PATH, LOGS_PATH, OUTPUT_PATH, MODELS_PATH
import numpy as np
import contextlib
import io
import os
import argparse
import json
import os
import random
import subprocess
from pathlib import Path
import uuid
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--concept', required=True)
parser.add_argument('-f', '--file', required=True)
parser.add_argument('-d', '--duration', required=True)
args=parser.parse_args()

def split_song(filepath, file_length, dest):
    command = [
        "ffmpeg",
        "-i", filepath,
        "-f", "segment",
        "-segment_time", str(file_length),
        "-c", "copy",
        "-reset_timestamps", "1",
        str(dest / "music_p%d.wav")
    ]
    subprocess.run(command, check=True)

def clean_directory(directory):
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)

if __name__ == '__main__':
    temp_dir = Path(INPUT_PATH('textual-inversion-v3', "temp"))
    clean_directory(temp_dir)

    split_song(args.file, args.duration, temp_dir)
    train_dir = Path(INPUT_PATH('textual-inversion-v3', "data", 'train', args.concept, 'audio'))
    clean_directory(train_dir)
    val_dir = Path(INPUT_PATH('textual-inversion-v3', "data", 'valid', args.concept, 'audio'))
    clean_directory(val_dir)

    segments = list(temp_dir.glob("music_p*.wav"))
    random.shuffle(segments)

    split_point = int(0.8 * len(segments))
    train_segments = segments[:split_point]
    valid_segments = segments[split_point:]
    
    for segment in train_segments:
        segment.rename(train_dir / segment.name)
    for segment in valid_segments:
        segment.rename(val_dir / segment.name)

    temp_dir.rmdir()
    print(f"Segments saved to {train_dir} and {val_dir}")

