from musicgen.data import (
    ConceptDataModule,
    get_ds,
    Concept,
)
from musicgen.model import ModelConfig, TransformerTextualInversion
from musicgen.data_const import Datasets
import pytorch_lightning as L
import torch
from audiocraft.models import MusicGen
import os
from tools.project import INPUT_PATH
import numpy as np
import gradio as gr

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    awailable_concepts = os.listdir(INPUT_PATH("textual-inversion-v3", "data", "train"))

    def train_concepts(concepts):
        ds_name = Datasets.TEXTUAL_INVERSION_V3
        cfg = ModelConfig(10, concepts=concepts, examples_len=1)
        ds = get_ds(ds_name).filter(lambda x: x["concept"] in cfg.concepts)
        model_name = f"facebook/musicgen-{cfg.model_name}"
        music_model = MusicGen.get_pretrained(model_name, device=DEVICE)
        music_model.set_generation_params(
            use_sampling=True,
            top_k=250,
            duration=cfg.examples_len,
            cfg_coef=cfg.cfg_coef,
        )
        model = TransformerTextualInversion.from_musicgen(music_model, cfg)
        dm = ConceptDataModule(ds, model.model.db, base_dir=ds_name, with_valid=False)
        trainer = L.Trainer(
            callbacks=[],
            enable_checkpointing=False,
            log_every_n_steps=10,
            max_epochs=10,
            accelerator=DEVICE,
        )
        trainer.fit(model, dm)
        return model

    def generate_music(model: TransformerTextualInversion, prompt_template):
        with torch.no_grad():
            prompts = []

            def gen_prompt(concept: Concept):
                prompts.append(prompt_template % concept.pseudoword())

            model.model.db.execute(gen_prompt)
            results = model.model.model.generate(prompts).cpu()
            results = results / np.max(np.abs(results.numpy()))
        if results.ndim == 3 and results.shape[0] > 1:
            audio_data = results[0]  # shape: (channels, samples)
        else:
            # Single sample scenario
            audio_data = results
        return (model.model.model.cfg.sample_rate, audio_data.numpy())

    with gr.Blocks() as demo:
        gr.Markdown("# MusicGen Textual Inversion Training & Generation")

        model_state = gr.State()
        concept_selector = gr.CheckboxGroup(
            label="Select Concepts",
            choices=awailable_concepts,
            value=awailable_concepts[:1],
        )

        train_button = gr.Button("Start Training")
        train_button.click(
            fn=train_concepts, inputs=concept_selector, outputs=[model_state]
        )

        prompt_text = gr.Textbox(
            label="Generation Prompt", value="High quality music in the style of %s"
        )
        generate_button = gr.Button("Generate Audio")
        audio_out = gr.Audio(label="Generated Audio", interactive=False)

        generate_button.click(
            fn=generate_music, inputs=[model_state, prompt_text], outputs=audio_out
        )
    demo.launch()
