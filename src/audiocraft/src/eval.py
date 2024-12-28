import numpy as np
import faiss
import msclap
from msclap import CLAP
from tools.project import INPUT_PATH, LOGS_PATH, OUTPUT_PATH, MODELS_PATH
from toolz import partition_all, concat
import os
import torch
from audioldm_eval.metrics.fad import FrechetAudioDistance
from argparse import ArgumentParser
import json
import tqdm
import contextlib
import io

DATASET = "concepts-dataset"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clap_model = CLAP(version="2023", use_cuda=torch.cuda.is_available())
fad_model = FrechetAudioDistance(verbose=False, use_pca=True, use_activation=True)

parser = ArgumentParser(add_help=False)
parser.add_argument("--other", type=str, default="musicgen-style")
parser.add_argument("--concepts", nargs="+", default=["8bit"])


@torch.no_grad
def get_dir_embeds(dir_path: str, recalc=True):
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


def kncc(train_embeds, val_embeds, gen_embeds, K=5):
    print("KNCC calc")
    index = faiss.IndexFlatIP(train_embeds.shape[-1])
    index.add(train_embeds)
    distances_val, indices_val = index.search(val_embeds, K)
    distances_gen, indices_gen = index.search(gen_embeds, K)
    res = 0.0
    for i in range(len(indices_val)):
        for j in range(len(indices_gen)):
            res += len(set(indices_val[i]).intersection(indices_gen[j])) / K
    return res / (i * j)


def knco(train_embeds, val_embeds, gen_embeds, K=5):
    print("KNCCO calc")
    index = faiss.IndexFlatIP(train_embeds.shape[-1])
    n = index.ntotal
    index.add(gen_embeds)
    new_ids = set(np.arange(n, n + gen_embeds.shape[0]))
    index.add(train_embeds)
    distances_val, indices_val = index.search(val_embeds, K)
    res = 0.0
    for ids in indices_val:
        res += len(new_ids.intersection(ids)) > 0
    return res / len(indices_val)


def fad(reference_path, examples_path):
    print("FAD calc")
    with contextlib.redirect_stdout(io.StringIO()):
        fd_score = fad_model.score(reference_path, examples_path, recalculate=True)
        if isinstance(fd_score, int):
            return float("inf")
        return list(fd_score.values())[0] * 1e-5


@torch.no_grad
def clap_sim(description, path):
    print("CLAP calc")
    embeds = get_dir_embeds(path)
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


def load_descriptions():
    with open(INPUT_PATH(DATASET, "metadata_concepts.json"), "r") as fh:
        return json.load(fh)


if __name__ == "__main__":
    args = parser.parse_args()
    result = {}
    descriptions = load_descriptions()
    for concept in tqdm.tqdm(args.concepts):
        fad_path = INPUT_PATH(DATASET, "data", "valid", concept, "fad")
        train_path = INPUT_PATH(DATASET, "data", "train", concept, "audio")
        ti_path = OUTPUT_PATH("musicgen-ti-generated", concept)
        other_path = OUTPUT_PATH(args.other, concept)
        train_embeddings = get_dir_embeds(train_path).numpy()
        val_embeddings = get_dir_embeds(fad_path).numpy()
        gen_embeddings = get_dir_embeds(ti_path).numpy()
        other_embeddings = get_dir_embeds(other_path).numpy()

        clap_ti = clap_sim(descriptions[concept], ti_path)
        clap_other = clap_sim(descriptions[concept], other_path)
        kncc_res_ti = kncc(train_embeddings, val_embeddings, gen_embeddings)
        knco_res_ti = knco(train_embeddings, val_embeddings, gen_embeddings)
        kncc_res_other = kncc(train_embeddings, val_embeddings, other_embeddings)
        knco_res_other = knco(train_embeddings, val_embeddings, other_embeddings)
        result[concept] = {
            "fad_ti": fad(fad_path, ti_path),
            "fad_other": fad(fad_path, other_path),
            "clap_ti": clap_ti,
            "clap_other": clap_other,
            "kncc_ti": kncc_res_ti,
            "knco_ti": kncc_res_ti,
            "kncc_other": kncc_res_other,
            "knco_other": knco_res_other,
        }
    res_name = f"ti-{args.other}.json"
    with open(OUTPUT_PATH("comparison", res_name), "w") as fh:
        json.dump(result, fh, indent=4)
