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
MAX_EPOCHS = 140

class StopTrainingCallback(L.Callback):
    def __init__(self, stop_flag):
        super().__init__()
        self.stop_flag = stop_flag

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self.stop_flag["value"] is True:
            trainer.should_stop = True

if __name__ == "__main__":
    awailable_concepts = os.listdir(INPUT_PATH("textual-inversion-v3", "data", "train"))

    def train_concepts(concepts, stop_flag):
        stop_flag["value"] = False
        ds_name = Datasets.TEXTUAL_INVERSION_V3
        cfg = ModelConfig(20, concepts=concepts, examples_len=5, ortho_alpha=1e-1, examples_num=400, model_name="medium")
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
        stop_callback = StopTrainingCallback(stop_flag=stop_flag)
        trainer = L.Trainer(
            callbacks=[stop_callback],
            enable_checkpointing=False,
            log_every_n_steps=10,
            max_epochs=MAX_EPOCHS,
            accelerator=DEVICE,
        )
        trainer.fit(model, dm)
        return model

    def generate_music(model: TransformerTextualInversion, prompt_template):
        with torch.no_grad():
            res = {'sr': model.model.model.cfg.sample_rate, 'data': {}}
            def gen_prompt(concept: Concept):
                prompt = prompt_template % concept.pseudoword()
                results = model.model.model.generate([prompt]*3).cpu()
                results = results / np.max(np.abs(results.numpy()))
                res['data'][concept.name] = results

            model.model.db.execute(gen_prompt)
        print("Generation completed")
        return res


    def stop_training(stop_flag):
        stop_flag["value"] = True

    with gr.Blocks() as demo:
        gr.Markdown("# MusicGen Textual Inversion Training & Generation")

        model_state = gr.State()
        stop_flag = gr.State({"value": False})
        generation = gr.State()
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    concept_selector = gr.CheckboxGroup(
                        label="Select Concepts",
                        choices=awailable_concepts,
                        value=awailable_concepts[:1],
                    )

                    train_button = gr.Button("Start Training")
                    stop_button = gr.Button("Stop Training")
                    train_button.click(
                        fn=train_concepts, inputs=[concept_selector, stop_flag], outputs=[model_state]
                    )
                    stop_button.click(
                        fn=stop_training,
                        inputs=[stop_flag]
                    )

                    prompt_text = gr.Textbox(
                        label="Generation Prompt", value="High quality music in the style of %s"
                    )

                    generate_button = gr.Button("Generate Audio")

                    generate_button.click(
                        fn=generate_music, inputs=[model_state, prompt_text], outputs=[generation]
                    )
            with gr.Row():
                with gr.Column():
                    @gr.render(inputs=[generation], triggers=[generation.change])
                    def render_content(content):
                        if not content:
                            return gr.Markdown("No content generated yet.")
                        sr = content['sr']
                        for name, audio in content['data'].items():
                            gr.Markdown(f"### {name}")
                            for a_idx in range(audio.shape[0]):
                                gr.Audio(value=(sr, audio[a_idx].squeeze().numpy()), label=f'{name} {a_idx}')

        
    demo.launch()
