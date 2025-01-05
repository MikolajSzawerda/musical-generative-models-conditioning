from tools.project import INPUT_PATH, LOGS_PATH, OUTPUT_PATH, RAW_PATH
import torch
from audiocraft.data.audio import audio_read, audio_write
from audiocraft.data.audio_utils import convert_audio_channels, convert_audio
import numpy as np
from audioldm_eval.metrics.fad import FrechetAudioDistance
import os
import sys
from fadtk.fad import FrechetAudioDistance
from fadtk.model_loader import CLAPLaionModel, VGGishModel
from fadtk.fad_batch import _cache_embedding_batch
from audiocraft.models import MusicGen
import shutil
import contextlib
import io
import warnings
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXAMPLES_LEN = 5
torch.cuda.is_available()


# model = CLAPLaionModel('music')


def _suppress_output():
    devnull = open(os.devnull, "w")
    sys.stdout = devnull
    sys.stderr = devnull


def _process_concept(concept: str, path: str):
    print(f"Starting concept: {concept}")
    model = VGGishModel()
    fad = FrechetAudioDistance(model, audio_load_worker=8, load_model=True)

    for f in Path(path).glob("*.*"):
        fad.cache_embedding_file(f)
    score = fad.score("fma_pop", path)

    shutil.rmtree(os.path.join(path, "embeddings"))
    shutil.rmtree(os.path.join(path, "convert"))
    shutil.rmtree(os.path.join(path, "stats"))
    return concept, score


def calc_fad(base_dir: str, concepts: list[str]) -> dict[str, float]:
    multiprocessing.set_start_method("spawn", force=True)
    with ProcessPoolExecutor(initializer=_suppress_output, max_workers=4) as executor:
        results = list(executor.map(_process_concept, concepts, [
        OUTPUT_PATH(base_dir, concept, "temp") for concept in concepts
    ]))
    return dict(results)


if __name__ == "__main__":
    print(calc_fad("textual-inversion-v3"))

# # score = fad.score('fma_pop', eval_dir)
# # print(score)
