from tools.project import INPUT_PATH, LOGS_PATH, OUTPUT_PATH, MODELS_PATH

from audiocraft.models import MusicGen
import torch
import tqdm
import torch
from audiocraft.data.audio import audio_read, audio_write
import numpy as np
import os
import random
from data import TokensProvider
import tqdm
from argparse import ArgumentParser


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROMPT = "In the style of %s"
DATASET = "concepts-dataset"

parser = ArgumentParser(add_help=False)
parser.add_argument("--model", type=str, default="ti")
parser.add_argument("--duration", type=int, default=5)
parser.add_argument("--num", type=int, default=10)
parser.add_argument("--run-name", type=str, default="unknown")
parser.add_argument("--model-name", type=str, default="small")
parser.add_argument("--ds-name", type=str, default=DATASET)
parser.add_argument("--out-dir", type=str, default="musicgen-ti-generated")
parser.add_argument("--concepts", nargs="+", default=["8bit"])


@torch.no_grad
def gen_ti(
    run_name,
    concepts,
    model_name,
    duration=5,
    num=10,
    ds_name=DATASET,
    out_dir="musicgen-ti-generated",
):
    text_emb = torch.load(MODELS_PATH(ds_name, f"{run_name}-best.pt"))
    print("Loading MusicGen")
    model = MusicGen.get_pretrained(f"facebook/musicgen-{model_name}")
    model.set_generation_params(use_sampling=True, top_k=250, duration=duration)

    print("Loading embedings")
    tokens_provider = TokensProvider(list(text_emb.items())[0][1]["embeds"].shape[0])
    text_conditioner = list(model.lm.condition_provider.conditioners.values())[0]
    tokenizer = text_conditioner.t5_tokenizer
    text_model = text_conditioner.t5
    for concept in text_emb.keys():
        tokenizer.add_tokens(tokens_provider.get(concept))
    text_model.resize_token_embeddings(len(tokenizer))
    for concept, data in text_emb.items():
        print(f"Loaded embeds for {concept} from {data['epoch']} epoch")
        for i, token in enumerate(tokens_provider.get(concept)):
            idx = tokenizer.convert_tokens_to_ids([token])[0]
            text_model.shared.weight[idx] = data["embeds"][i]

    print("Started generation")
    for concept in tqdm.tqdm(concepts):
        prompt = PROMPT % tokens_provider.get_str(concept)
        res = model.generate([prompt] * num, progress=True)
        for a_idx in range(res.shape[0]):
            music = res[a_idx].cpu()
            music = music / np.max(np.abs(music.numpy()))
            path = OUTPUT_PATH(out_dir, concept, f"music_p{a_idx}")
            audio_write(path, music, model.cfg.sample_rate)


def gen_style(concepts, duration=5, num=10, ds_name=DATASET):
    model = MusicGen.get_pretrained("facebook/musicgen-style")
    model.set_generation_params(
        duration=duration,
        use_sampling=True,
        top_k=250,
        cfg_coef=3.0,  # Classifier Free Guidance coefficient
        cfg_coef_beta=8.0,  # double CFG is necessary for text-and-style conditioning
        # Beta in the double CFG formula. between 1 and 9. When set to 1 it is equivalent to normal CFG.
        # When we increase this parameter, the text condition is pushed. See the bottom of https://musicgenstyle.github.io/
        # to better understand the effects of the double CFG coefficients.
    )

    model.set_style_conditioner_params(
        eval_q=2,  # integer between 1 and 6
        # eval_q is the level of quantization that passes
        # through the conditioner. When low, the models adheres less to the
        # audio conditioning
        excerpt_length=4.5,  # the length in seconds that is taken by the model in the provided excerpt, can be
        # between 1.5 and 4.5 seconds but it has to be shortest to the length of the provided conditioning
    )
    for concept in tqdm.tqdm(concepts):
        examples = os.listdir(INPUT_PATH(ds_name, "data", "valid", f"{concept}", "fad"))
        random.shuffle(examples)
        songs = []
        for fname in tqdm.tqdm(random.choices(examples, k=num)):
            melody, sr = audio_read(
                INPUT_PATH(ds_name, "data", "valid", f"{concept}", "fad", fname),
                pad=True,
                duration=5,
            )
            songs.append(melody[0][None].expand(1, -1, -1))
        songs = torch.cat(songs, dim=0)
        results = model.generate_with_chroma(
            [None] * len(songs), songs, sr, progress=True
        )
        for a_idx in range(results.shape[0]):
            music = results[a_idx].cpu()
            music = music / np.max(np.abs(music.numpy()))
            path = OUTPUT_PATH("musicgen-style", concept, f"music_p{a_idx}")
            audio_write(path, music, model.cfg.sample_rate)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.model == "ti":
        gen_ti(
            args.run_name,
            args.concepts,
            args.model_name,
            args.duration,
            args.num,
            args.ds_name,
            args.out_dir,
        )
    elif args.model == "style":
        gen_style(args.concepts, args.duration, args.num, args.ds_name)
    else:
        print("Unsupported model to generation")
