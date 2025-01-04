from tools.project import INPUT_PATH, LOGS_PATH, OUTPUT_PATH, RAW_PATH
import torch
from audiocraft.data.audio import audio_read, audio_write
from audiocraft.data.audio_utils import convert_audio_channels, convert_audio
import numpy as np
from audioldm_eval.metrics.fad import FrechetAudioDistance
import os
import sys
from fadtk.fad import FrechetAudioDistance, log
from fadtk.model_loader import CLAPLaionModel, VGGishModel
from fadtk.fad_batch import cache_embedding_files
from audiocraft.models import MusicGen
import shutil
import contextlib
import io
import warnings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
EXAMPLES_LEN = 5
torch.cuda.is_available()

# model = CLAPLaionModel('music')
model = VGGishModel()
eval_dir = OUTPUT_PATH('concepts-dataset', '8bit-slow', 'temp')
cache_embedding_files('fma_pop', model)
cache_embedding_files(eval_dir, model)
fad = FrechetAudioDistance(model, audio_load_worker=8, load_model=False)
fad.score('fma_pop', eval_dir)

cache_embedding_files(OUTPUT_PATH("textual-inversion", 'metal', 'temp'), model)
score = fad.score('fma_pop', OUTPUT_PATH("textual-inversion", 'metal', 'temp'))
shutil.rmtree(os.path.join(OUTPUT_PATH("textual-inversion", 'metal', 'temp'), 'embeddings'))
shutil.rmtree(os.path.join(OUTPUT_PATH("textual-inversion", 'metal', 'temp'), 'convert'))
shutil.rmtree(os.path.join(OUTPUT_PATH("textual-inversion", 'metal', 'temp'), 'stats'))