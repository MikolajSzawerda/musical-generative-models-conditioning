import numpy as np
import faiss
import msclap
from msclap import CLAP
from tools.project import INPUT_PATH, LOGS_PATH, OUTPUT_PATH, MODELS_PATH
from toolz import partition_all, concat
import os
import torch
# from audioldm_eval.metrics.fad import FrechetAudioDistance
from argparse import ArgumentParser
import json
import tqdm
import contextlib
import io
from metrics import _process_concept
from fadtk.model_loader import CLAPLaionModel
from fadtk.fad import FrechetAudioDistance
from fadtk.utils import get_cache_embedding_path
from pathlib import Path
DATASET = "concepts-dataset"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clap_model = CLAP(version="2023", use_cuda=torch.cuda.is_available())
model = CLAPLaionModel('music')
fad = FrechetAudioDistance(model)
# fad_model = FrechetAudioDistance(verbose=False, use_pca=True, use_activation=True)

parser = ArgumentParser(add_help=False)
parser.add_argument("--other", type=str, default="musicgen-style")
parser.add_argument("--concepts", nargs="+", default=["8bit"])
parser.add_argument("--out", type=str, default="ti-musicgen-style.json")


@torch.no_grad
def get_dir_embeds(dir_path: str, recalc=True):
    """
    Calculate and retrieve directory embeddings for audio files. This function checks
    if precomputed embeddings are cached. If caching is available and recalculation
    is not requested, the function loads the cached embeddings. Otherwise, it processes
    the files in the directory, computes the embeddings in batches, and caches the
    result for future use.

    :param dir_path:
        The directory path containing audio files to process.
    :param recalc:
        A boolean indicating whether to force recalculation of embeddings, even if a
        cached file exists. Defaults to ``True``.
    :return:
        A tensor containing the computed or loaded embeddings of the audio files.
    """
    dir_name = os.path.basename(dir_path)
    cache_path = os.path.join(os.path.dirname(dir_path), f"clap_feature_{dir_name}.pt")
    if os.path.exists(cache_path) and not recalc:
        return torch.load(cache_path)
    files = [f for f in os.listdir(dir_path) if not os.path.isdir(os.path.join(dir_path, f))]
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
    """
    Compute the KNCC score by finding the overlap
    of nearest neighbors between validation embeddings and generated embeddings.

    The function uses FAISS to construct an index for nearest neighbor search with
    the provided training embeddings. The nearest neighbors for both validation
    and generated embeddings are computed, and the KNCC score is calculated by
    determining the fraction of overlapping neighbors normalized by the total
    number of pairs.

    :param train_embeds: Training embeddings, expected as a NumPy array or similar
        object with shape (n_samples, n_features).
    :param val_embeds: Validation embeddings, expected as a NumPy array or similar
        object with shape (m_samples, n_features).
    :param gen_embeds: Generated embeddings, expected as a NumPy array or similar
        object with shape (k_samples, n_features).
    :param K: Number of nearest neighbors to consider for overlap computation.
        Defaults to 5.
    :return: Proportion of overlapping neighbors between validation and generated
        embeddings. The return value is a float representing the KNCC score.
    :rtype: float
    """
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
    """
    Calculates the fraction of validation embeddings (`val_embeds`) for which at least one of the closest
    `K` neighbors, identified within the combined corpus of generated embeddings (`gen_embeds`) and
    training embeddings (`train_embeds`), belongs exclusively to the set of generated embeddings.

    This function leverages the FAISS library to perform high-dimensional nearest neighbor search based
    on inner product similarities. It ensures that the neighborhood around the validation embeddings is
    augmented by both generated and training embeddings but only checks for exclusive membership
    in the set of generated embeddings.

    :param train_embeds: High-dimensional embeddings from the training dataset used to populate the
        search index.
        Type: numpy.ndarray of shape `(num_train_samples, embed_dim)`

    :param val_embeds: High-dimensional embeddings from the validation dataset whose nearest
        neighbors are queried.
        Type: numpy.ndarray of shape `(num_val_samples, embed_dim)`

    :param gen_embeds: High-dimensional embeddings generated by some process, utilized for checking
        exclusive neighborhood membership.
        Type: numpy.ndarray of shape `(num_generated_samples, embed_dim)`

    :param K: Number of nearest neighbors to consider in the search for each validation embedding.
        Defaults to 5.
        Type: int

    :return: Fraction of validation examples for which at least one of their `K` nearest neighbors is an
        embedding exclusively from the generated embedding set.
        Type: float
    """
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


# def fad(reference_path, examples_path):
#     print("FAD calc")
#     with contextlib.redirect_stdout(io.StringIO()):
#         fd_score = fad_model.score(reference_path, examples_path, recalculate=True)
#         if isinstance(fd_score, int):
#             return float("inf")
#         return list(fd_score.values())[0] * 1e-5


@torch.no_grad
def clap_sim(description, path):
    """
    Compute the similarity between a given description and embeddings obtained from a directory
    using a pre-trained CLAP model.

    This function calculates the similarity between text embeddings derived from the provided
    description and embeddings for audio/image files located in the specified directory. It uses
    a Contrastive Language-Audio Pretraining (CLAP) model for the computation. The function is
    decorated with `@torch.no_grad`, which ensures no gradients are computed during execution,
    thereby improving performance and resource efficiency when used in inferencing scenarios.

    :param description: The textual description for which text embeddings will be generated
        to compute similarity.
    :type description: str
    :param path: The file path to the directory containing audio/image files, from which
        embeddings will be extracted for similarity computation.
    :type path: str
    :return: The mean similarity score indicating how closely the provided description matches
        the embeddings from files in the specified directory.
    :rtype: float
    """
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
    """
    Loads and returns the descriptions from a JSON file.

    This function reads a JSON file called "metadata_concepts.json" located
    in the input path of the dataset and parses its contents. The parsed
    data, assumed to be a dictionary of metadata descriptions, is then
    returned to the caller. The input path is dynamically constructed
    using the provided `INPUT_PATH` function.

    :return: Parsed dictionary containing the descriptions from the
        "metadata_concepts.json" file.
    :rtype: dict
    """
    with open(INPUT_PATH(DATASET, "metadata_concepts.json"), "r") as fh:
        return json.load(fh)



def clap_fad(ref_path, eval_path):
    """
    Calculates the Fréchet Audio Distance (FAD) score between reference and evaluation datasets.
    This function first ensures the embedding files required for the calculations
    are cached for both datasets. If the embeddings already exist at the specified
    paths, it skips recalculating and caching them. After confirming cached embeddings
    are available, it computes and returns the Fréchet Audio Distance score.

    :param ref_path: Path to the directory containing the reference dataset files.
    :type ref_path: str
    :param eval_path: Path to the directory containing the evaluation dataset files.
    :type eval_path: str
    :return: The computed Fréchet Audio Distance score.
    :rtype: float
    """
    def cache_path(path):
        if os.path.exists(os.path.join(path, 'embeddings', model.name)):
            return
        
        for f in Path(path).glob("*.*"):
            if os.path.isdir(f):
                continue
            fad.cache_embedding_file(f)
    cache_path(ref_path)
    cache_path(eval_path)
    return fad.score(ref_path, eval_path)

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
        clap_base = clap_sim(descriptions[concept], train_path)
        clap_other = clap_sim(descriptions[concept], other_path)

        fad_clap_ti = clap_fad(train_path, ti_path)
        fad_clap_other = clap_fad(train_path, other_path)

        kncc_res_ti = kncc(train_embeddings, val_embeddings, gen_embeddings)
        knco_res_ti = knco(train_embeddings, val_embeddings, gen_embeddings)
        kncc_res_other = kncc(train_embeddings, val_embeddings, other_embeddings)
        knco_res_other = knco(train_embeddings, val_embeddings, other_embeddings)
        result[concept] = {
            "fad_base": _process_concept(concept, train_path, clear=False)[1],
            "fad_ti": _process_concept(concept, ti_path, clear=False)[1],
            "fad_other": _process_concept(concept, other_path, clear=False)[1],
            "clap_base": clap_base,
            "clap_ti": clap_ti,
            "clap_other": clap_other,
            "fad_clap_ti": fad_clap_ti,
            "fad_clap_other": fad_clap_other,
            "kncc_ti": kncc_res_ti,
            "knco_ti": kncc_res_ti,
            "kncc_other": kncc_res_other,
            "knco_other": knco_res_other,
        }
    with open(OUTPUT_PATH("comparison", args.out), "w") as fh:
        json.dump(result, fh, indent=4)
