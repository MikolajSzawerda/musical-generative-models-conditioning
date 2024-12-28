import tqdm
from tools.project import INPUT_PATH, LOGS_PATH, OUTPUT_PATH, MODELS_PATH
import numpy as np
import contextlib
import io
import os
import argparse
import json
import random
import subprocess
from pathlib import Path
import uuid
import shutil

DATASET = "eval-ds"

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--concept', required=True)
parser.add_argument('-f', '--file', default="")
parser.add_argument('--directory', default="")
parser.add_argument('-d', '--duration', required=True)
parser.add_argument("--no-valid", action="store_true")
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
    def handle_file(file_path):
        temp_dir = Path(INPUT_PATH(DATASET, "temp"))
        clean_directory(temp_dir)

        split_song(file_path, args.duration, temp_dir)
        train_dir = Path(INPUT_PATH(DATASET, "data", 'train', args.concept, 'audio'))
        clean_directory(train_dir)
        val_dir = Path(INPUT_PATH(DATASET, "data", 'valid', args.concept, 'audio'))
        clean_directory(val_dir)

        segments = list(temp_dir.glob("music_p*.wav"))
        random.shuffle(segments)
        if not args.no_valid:
            split_point = int(0.8 * len(segments))
            train_segments = segments[:split_point]
            valid_segments = segments[split_point:]
            for segment in valid_segments:
                segment.rename(val_dir / segment.name)
        else:
            train_segments = segments
        
        for segment in train_segments:
            segment.rename(train_dir / segment.name)
        

        temp_dir.rmdir()
        print(f"Segments saved to {train_dir} and {val_dir}")
    if args.directory != '':
        files = os.listdir(args.directory)
        for f in tqdm.tqdm(files):
            handle_file(os.path.join(args.directory, f))
    elif args.file != '':
        handle_file(args.file)
    else:
        print("No path provided")

