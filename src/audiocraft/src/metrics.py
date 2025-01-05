import json
import multiprocessing
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from audioldm_eval.metrics.fad import FrechetAudioDistance
from fadtk.fad import FrechetAudioDistance
from fadtk.model_loader import VGGishModel
from msclap import CLAP
from tools.project import INPUT_PATH, OUTPUT_PATH
from toolz import partition_all, concat

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXAMPLES_LEN = 5
torch.cuda.is_available()


# model = CLAPLaionModel('music')


def _suppress_output():
    devnull = open(os.devnull, "w")
    sys.stdout = devnull
    sys.stderr = devnull


def _process_concept(concept: str, path: str, clear=True):
    print(f"Starting concept: {concept}")
    model = VGGishModel()
    fad = FrechetAudioDistance(model, audio_load_worker=8, load_model=True)

    for f in Path(path).glob("*.*"):
        fad.cache_embedding_file(f)
    score = fad.score("fma_pop", path)
    if clear:
        shutil.rmtree(os.path.join(path, "embeddings"))
        shutil.rmtree(os.path.join(path, "convert"))
        shutil.rmtree(os.path.join(path, "stats"))
    return concept, score


def calc_fad(base_dir: str, concepts: list[str]) -> dict[str, float]:
    multiprocessing.set_start_method("spawn", force=True)
    with ProcessPoolExecutor(initializer=_suppress_output, max_workers=4) as executor:
        results = list(
            executor.map(
                _process_concept,
                concepts,
                [OUTPUT_PATH(base_dir, concept, "temp") for concept in concepts],
            )
        )
    return dict(results)


@torch.no_grad()
def get_dir_embeds(clap_model: CLAP, dir_path: str, recalc=True):
    dir_name = os.path.basename(dir_path)
    cache_path = os.path.join(os.path.dirname(dir_path), f"clap_feature_{dir_name}.pt")
    if os.path.exists(cache_path) and not recalc:
        return torch.load(cache_path)
    files = os.listdir(dir_path)
    batches = partition_all(20, files)
    res = []

    def get_embs(paths):
        return clap_model.get_audio_embeddings(paths)

    for batch in batches:
        res.append(get_embs(os.path.join(dir_path, f) for f in batch))
    res = torch.stack(list(concat(res))).detach().cpu()
    torch.save(res, cache_path)
    return res


@torch.no_grad()
def clap_sim(clap_model: CLAP, description, path):
    print("CLAP calc")
    embeds = get_dir_embeds(clap_model, path)
    text_embeds = clap_model.get_text_embeddings([description]).expand(
        embeds.shape[0], -1
    )
    return (
        clap_model.compute_similarity(embeds.to("cuda"), text_embeds)[:, 0]
        .mean(dim=0)
        .detach()
        .cpu()
        .item()
    )


def _calc_clap(concept, path, description):
    clap_model = CLAP(version="2023", use_cuda=torch.cuda.is_available())
    return concept, clap_sim(clap_model, description, path)


def calc_clap(base_dir: str, concepts_desc: dict[str, str]) -> dict[str, float]:
    multiprocessing.set_start_method("spawn", force=True)
    with ProcessPoolExecutor(initializer=_suppress_output, max_workers=1) as executor:
        results = list(
            executor.map(
                _calc_clap,
                concepts_desc.keys(),
                [
                    OUTPUT_PATH(base_dir, concept, "temp")
                    for concept in concepts_desc.keys()
                ],
                concepts_desc.values(),
            )
        )
    return dict(results)


if __name__ == "__main__":
    base_dir = "textual-inversion-v3"
    with open(INPUT_PATH(base_dir, "metadata_concepts.json"), "r") as fh:
        desc = json.load(fh)
    print(
        calc_clap(
            base_dir, {k: v for k, v in desc.items() if k in ["cluster_0", "cluster_1"]}
        )
    )

# # score = fad.score('fma_pop', eval_dir)
# # print(score)
