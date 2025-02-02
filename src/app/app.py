from argparse import ArgumentParser

from datasets import DatasetDict, load_dataset
import tempfile
from musicgen.data import (
    ConceptDataModule,
    Concept,
    resample_ds,
    ConceptEmbeds,
    TextConcepts,
)
from musicgen.model import ModelConfig, TransformerTextualInversion
import pytorch_lightning as L
import torch
from audiocraft.models import MusicGen
import os
import numpy as np
import gradio as gr
from pathlib import Path
import threading
import time
import logging
import plotly.graph_objects as go
import petname
from fadtk.model_loader import CLAPLaionModel
from fadtk.fad import FrechetAudioDistance, calc_frechet_distance
from fadtk.utils import get_cache_embedding_path
from toolz import partition_all
from musicgen.utils import suppress_all_output
from audiocraft.data.audio import audio_write

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_EPOCHS = 140
logger = logging.getLogger(__name__)


class StopTrainingCallback(L.Callback):
    def __init__(self, stop_flag):
        super().__init__()
        self.stop_flag = stop_flag
        self.current_epoch = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self.current_epoch = trainer.current_epoch

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.stop_flag["value"] is True:
            trainer.should_stop = True


class SaveEmbeddingsCallback(L.Callback):
    def __init__(
        self,
        base_dir: Path,
        run_name: str,
        concepts: dict[str, Concept],
        weights: torch.Tensor,
    ):
        super().__init__()
        self.base_dir = base_dir
        self.run_name = run_name
        self.concepts = concepts
        # self.best_score = {c: float("inf") for c in cfg.concepts.concepts.keys()}
        self.weights = weights
        self.best_embeds = {
            c.name: weights[c.token_ids].detach().cpu() for c in concepts.values()
        }

    def on_train_epoch_end(self, trainer, pl_module):
        # if trainer.current_epoch % self.cfg.n_epochs != 0:
        #     return

        def update(concept: Concept):
            logger.info(
                f"Updating saved embedings for {concept.name} at {trainer.current_epoch} epoch"
            )
            self.best_embeds[concept.name] = {
                "epoch": trainer.current_epoch,
                "embeds": self.weights[concept.token_ids].detach().cpu(),
            }

        for c in self.concepts.values():
            update(c)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.best_embeds, self.base_dir / f"{self.run_name}.pt")


class NoOpEvalCallback(L.Callback):
    def __init__(self):
        self.evaluation = {}
        self.calback_state = None


class EvalCallback(L.Callback):
    def __init__(
        self,
        base_dir: Path,
        concepts: TextConcepts,
        fad: FrechetAudioDistance,
        n_epochs: int = 10,
        n_gen: int = 5,
        prompt_template: str = "In the style of %s",
        generation_batch: int = 50,
    ):
        super().__init__()
        self.base_dir = base_dir
        self.fad = fad
        self.concepts = concepts
        self.n_epochs = n_epochs
        self.prompt_template = prompt_template
        self.n_gen = n_gen
        self.generation_batch = generation_batch
        self.evaluation = {c: [] for c in concepts.concepts_names}
        self.calback_state = None

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.n_epochs != 0:
            return
        self.calback_state = "Generating examples for evaluation"
        prompts = []

        def gen_prompt(c: Concept):
            prompts.extend(
                [
                    (
                        c.name,
                        self.prompt_template % c.pseudoword(),
                    )
                    for _ in range(self.n_gen)
                ]
            )

        self.concepts.execute(gen_prompt)
        prompts_batches = partition_all(self.generation_batch, prompts)
        concept_counter = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_dir = Path(tmpdir)

            def save_audio(audio: torch.Tensor, concept_name: str):
                ctn = concept_counter.get(concept_name, 0)
                path = tmp_dir / concept_name / "temp" / f"music_p{ctn}"
                audio_write(path, audio, pl_module.model.model.cfg.sample_rate)
                concept_counter[concept_name] = ctn + 1

            for batch in prompts_batches:
                concepts, prompts = list(zip(*batch))
                with torch.no_grad():
                    results = pl_module.model.model.generate(prompts).cpu()
                results = results / np.max(np.abs(results.numpy()))
                for concept_name, audio in zip(concepts, results):
                    save_audio(audio, concept_name)
            self.calback_state = "Evaluating examples"
            with suppress_all_output():
                for c_name in self.concepts.concepts_names:

                    def cache_path_emb(path: Path):
                        for f in path.glob("*.*"):
                            if get_cache_embedding_path(self.fad.ml.name, f).exists():
                                continue
                            self.fad.cache_embedding_file(f)

                    base_path = self.base_dir / "data" / "train" / c_name / "audio"
                    c_path = tmp_dir / c_name / "temp"
                    cache_path_emb(base_path)
                    cache_path_emb(c_path)

                    mu_gen, cov_gen = self.fad.load_stats(c_path)
                    mu_ref, cov_ref = self.fad.load_stats(base_path)
                    self.evaluation[c_name].append(
                        calc_frechet_distance(mu_ref, cov_ref, mu_gen, cov_gen)
                    )
            self.calback_state = None


def get_ds(ds_path: Path) -> DatasetDict:
    return load_dataset(
        "json",
        data_files={
            "train": str(ds_path / "metadata_train.json"),
        },
    )


default_cfg = {
    "model_name": "small",
    "tokens_num": 20,
    "examples_len": 5,
    "examples_num": 400,
}

generation_cfg = {"num_per_concept": 3, "examples_len": 5}

train_thread = None

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--embeds-dir", type=str, required=True)
    args = parser.parse_args()
    ds_path = Path(args.dataset)
    ds_name = os.path.basename(ds_path)
    awailable_concepts = os.listdir(ds_path / "data" / "train")

    models_path = Path(args.embeds_dir)

    def fetch_embeds():
        names = [file.name for file in models_path.glob("*.pt")]
        names.sort()
        return names

    def train_concepts(concepts, cfg_in, stop_flag, run_eval: bool):
        stop_flag["value"] = False
        curr_models_num = len(os.listdir(models_path)) + 1
        run_name = f"{curr_models_num}-{cfg_in['model_name']}-{cfg_in['tokens_num']}-{petname.Generate(2)}"
        cfg = ModelConfig(
            cfg_in["tokens_num"],
            concepts=concepts,
            examples_len=cfg_in["examples_len"],
            ortho_alpha=1e-1,
            examples_num=cfg_in["examples_num"],
            model_name=cfg_in["model_name"],
        )
        ds = get_ds(ds_path).filter(lambda x: x["concept"] in cfg.concepts)
        ds = resample_ds(ds, cfg.examples_num)
        model_name = f"facebook/musicgen-{cfg.model_name}"
        music_model = MusicGen.get_pretrained(model_name, device=DEVICE)
        music_model.set_generation_params(
            use_sampling=True,
            top_k=250,
            duration=cfg.examples_len,
            cfg_coef=cfg.cfg_coef,
        )
        model = TransformerTextualInversion.from_musicgen(music_model, cfg)

        dm = ConceptDataModule(ds, model.model.db, with_valid=False)
        stop_callback = StopTrainingCallback(stop_flag=stop_flag)
        save_callback = SaveEmbeddingsCallback(
            base_dir=models_path,
            run_name=run_name,
            concepts=model.model.db.db,
            weights=model.model.text_weights,
        )
        if run_eval:
            with suppress_all_output():
                clap = CLAPLaionModel("music")
                fad = FrechetAudioDistance(clap)
            eval_callback = EvalCallback(ds_path, model.model.db, fad)
        else:
            eval_callback = NoOpEvalCallback()
        trainer = L.Trainer(
            callbacks=[stop_callback, eval_callback, save_callback],
            enable_checkpointing=False,
            log_every_n_steps=10,
            max_epochs=MAX_EPOCHS,
            accelerator=DEVICE,
        )

        def run_train():
            logger.info("Training started")
            trainer.fit(model, dm)

        fig = go.Figure(layout=go.Layout(template="plotly_dark"))
        fig.update_layout(
            title=f"Training progress for run: {run_name}",
            showlegend=True,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        for label in eval_callback.evaluation.keys():
            fig.add_trace(go.Scatter(y=[], mode="lines+markers", name=label))
        global train_thread
        if train_thread is None or not train_thread.is_alive():
            logger.info("Starting training")
            train_thread = threading.Thread(target=run_train)
            train_thread.start()
        i = 0
        while train_thread.is_alive():
            current_epoch = stop_callback.current_epoch
            for trace_index, label in enumerate(eval_callback.evaluation.keys()):
                fig.data[trace_index].y = eval_callback.evaluation[label]
            yield model, eval_callback.calback_state or "Training in progress...", gr.update(
                value=current_epoch, visible=True
            ), gr.update(
                value=fig, visible=run_eval
            ), run_name
            time.sleep(1.0)
            i += 1

        logger.info("Training finished")
        yield model, "Training finished", i, fig, run_name

    def generate_music(
        concepts: list[str],
        loaded_embeds: dict[str, torch.Tensor],
        cfg_in: dict[str, any],
        generation_cfg: dict[str, any],
        prompt_template,
        progress=gr.Progress(track_tqdm=True),
    ):
        if len(concepts) == 0:
            logger.warning("No concepts to generate")
            return {"sr": 48000, "data": {}}
        logger.info("Starting generation")
        cfg = ModelConfig(
            cfg_in["tokens_num"],
            concepts=concepts,
            examples_len=generation_cfg["examples_len"],
            ortho_alpha=1e-1,
            model_name=cfg_in["model_name"],
        )
        model_name = f"facebook/musicgen-{cfg.model_name}"
        music_model = MusicGen.get_pretrained(model_name, device=DEVICE)
        music_model.set_generation_params(
            use_sampling=True,
            top_k=250,
            duration=cfg.examples_len,
            cfg_coef=cfg.cfg_coef,
        )
        embeds = {
            c: ConceptEmbeds(loaded_embeds[c]["epoch"], loaded_embeds[c]["embeds"])
            for c in concepts
        }
        model = TransformerTextualInversion.from_previous_run(embeds, music_model, cfg)
        with torch.no_grad():
            res = {"sr": model.model.model.cfg.sample_rate, "data": {}}

            def gen_prompt(concept: Concept):
                prompt = prompt_template % concept.pseudoword()
                results = model.model.model.generate(
                    [prompt] * generation_cfg["num_per_concept"]
                ).cpu()
                results = results / np.max(np.abs(results.numpy()))
                res["data"][concept.name] = results
                return res

            for concept in progress.tqdm(model.model.db.db.values()):
                yield "Generating...", gen_prompt(concept)
        logger.info("Generation completed")
        yield "Generated", res

    def stop_training(stop_flag):
        stop_flag["value"] = True

    with gr.Blocks() as demo:
        gr.Markdown("# MusicGen Textual Inversion Training & Generation")

        model_state = gr.State()
        stop_flag = gr.State({"value": False})
        generation = gr.State()
        model_cfg = gr.State(value=default_cfg)
        run_name_state = gr.State(value="")
        with gr.Column():
            gr.Markdown("## Configuration")
            model_name_input = gr.Dropdown(
                choices=["small", "medium", "large"], value="small", label="Model size"
            )
            model_name_input.change(
                fn=lambda val, cfg: cfg.update({"model_name": val}) or cfg,
                inputs=[model_name_input, model_cfg],
                outputs=[model_cfg],
            )

        with gr.Column():
            with gr.Tab("Train"):
                gr.Markdown("### Traning Configuration")
                tokens_num_input = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=20,
                    step=1,
                    label="Text Representation Size(# of tokens)",
                )
                tokens_num_input.change(
                    fn=lambda val, cfg: cfg.update({"tokens_num": int(val)}) or cfg,
                    inputs=[tokens_num_input, model_cfg],
                    outputs=[model_cfg],
                )
                examples_len_input = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=5,
                    label="Length of single example per concept(in seconds)",
                    step=1,
                )
                examples_len_input.change(
                    fn=lambda val, cfg: cfg.update({"examples_len": int(val)}) or cfg,
                    inputs=[examples_len_input, model_cfg],
                    outputs=[model_cfg],
                )
                examples_num_input = gr.Slider(
                    minimum=40,
                    maximum=800,
                    value=400,
                    step=10,
                    label="# of examples per concept",
                )
                examples_num_input.change(
                    fn=lambda val, cfg: cfg.update({"examples_num": int(val)}) or cfg,
                    inputs=[examples_num_input, model_cfg],
                    outputs=[model_cfg],
                )
                concept_selector = gr.CheckboxGroup(
                    label="Select Concepts",
                    choices=awailable_concepts,
                )

                train_button = gr.Button("Start Training")
                stop_button = gr.Button("Stop Training")
                training_status_output = gr.Text(
                    value="Waiting for training to start...", label=""
                )
                epoch_output = gr.Text("Epoch", label="Epoch", visible=False)
                run_eval_input = gr.Checkbox(
                    label="Run model evaluation during training"
                )
                training_plot = gr.Plot(visible=False)
                train_button.click(
                    fn=train_concepts,
                    inputs=[concept_selector, model_cfg, stop_flag, run_eval_input],
                    outputs=[
                        model_state,
                        training_status_output,
                        epoch_output,
                        training_plot,
                        run_name_state,
                    ],
                )
                stop_button.click(fn=stop_training, inputs=[stop_flag])

            with gr.Tab("Generate"):
                selected_embed = gr.State()
                loaded_embeding = gr.State()
                generation_cfg = gr.State(value=generation_cfg)

                gr.Markdown("### Generation Configuration")
                num_per_concept_input = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=3,
                    step=1,
                    label="Number of generated examples per concept",
                )
                num_per_concept_input.change(
                    fn=lambda val, cfg: cfg.update({"num_per_concept": int(val)})
                    or cfg,
                    inputs=[num_per_concept_input, generation_cfg],
                    outputs=[generation_cfg],
                )
                gen_examples_len_input = gr.Slider(
                    minimum=1,
                    maximum=30,
                    value=5,
                    label="Length of single example(in seconds)",
                    step=1,
                )
                gen_examples_len_input.change(
                    fn=lambda val, cfg: cfg.update({"examples_len": int(val)}) or cfg,
                    inputs=[gen_examples_len_input, generation_cfg],
                    outputs=[generation_cfg],
                )

                refresh_embeds = gr.Button("Refresh learned embeddings")
                file_dropdown = gr.Dropdown(
                    label="Select a File", interactive=True, choices=[]
                )
                refresh_embeds.click(
                    fn=lambda: gr.update(choices=fetch_embeds()),
                    outputs=[file_dropdown],
                )
                selected_concepts = gr.CheckboxGroup(
                    label="Select Concepts",
                    choices=[],
                    visible=True,
                    interactive=True,
                )

                def load_embeding(f_name):
                    f_path = str(models_path / f_name)
                    embeds = torch.load(f_path)
                    try:
                        concepts = list(embeds.keys())
                    except Exception:
                        concepts = []
                    return f_path, gr.update(choices=concepts, visible=True), embeds

                file_dropdown.change(
                    fn=load_embeding,
                    inputs=[file_dropdown],
                    outputs=[selected_embed, selected_concepts, loaded_embeding],
                )

                prompt_text = gr.Textbox(
                    label="Generation Prompt(put single %s where pseudowords should be added)",
                    value="High quality music in the style of %s",
                )

                generate_button = gr.Button("Generate Audio")
                generate_status = gr.Text("Waiting for generation...", label="")
                generate_button.click(
                    fn=generate_music,
                    inputs=[
                        selected_concepts,
                        loaded_embeding,
                        model_cfg,
                        generation_cfg,
                        prompt_text,
                    ],
                    outputs=[generate_status, generation],
                )

                @gr.render(inputs=[generation], triggers=[generation.change])
                def render_content(content):
                    if not content:
                        return gr.Markdown("No content generated yet.")
                    sr = content["sr"]
                    for name, audio in content["data"].items():
                        with gr.Accordion(name):
                            for a_idx in range(audio.shape[0]):
                                gr.Audio(
                                    value=(sr, audio[a_idx].squeeze().numpy()),
                                    label=f"{name} {a_idx}",
                                )

    demo.launch()
