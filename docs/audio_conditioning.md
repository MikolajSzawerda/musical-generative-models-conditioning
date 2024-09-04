> popular approach to use CLAP+non annotatted audio during training - no annotation needed to train

## Audio conditioning

https://arxiv.org/html/2407.12563v1
[musicgenstyle.github.io](https://arxiv.org/html/2407.12563#:~:text=musicgenstyle.github.io)

- conditioning via audio input
- usage of textual inversion to convert audio to pseudo words together with pretrained model
    - MusicGen as transformer decoder, init of textual embeding, then optimization of cross entropy(musicgen parameters, all previous song chunks, embeding) with respect to EMBEDING, embeding then fed to model
- trained from scratch music language model(text conditioner+quantized feature extractor) + usage of double classifier free guidance(balance of conditioning)
    - music gen+style conditioner - T5 text embeding+style embeding as prefix tokens to EnCodec of audio
- used metrics: Nearest Neighbours in Common(KNN), FAD

## Textual inversion

https://arxiv.org/pdf/2208.01618

given: textual description of audios, audios
- init embeding with originall embeding generator
- generate audio with model
- calculate loss between given audios and generated ones
- optimize with respect to embeding

## MusTango

https://arxiv.org/html/2311.08355
(creation of MusicBench dataset)

- diffusion model with domain music knowledge: text, chords, beats, tempo, key
- usage of MuNet - music knowledge unet
- usage od Latent Diffusion Model
    - creation of latent var using VAE+conditining(audio+text) - usage of AudioLDM to perform forward diffusion
    - usage of MuNet to denoise: cross-attention conditioning blocks: encoders of beat(beat type(up/down one-hot)+timing),chord(FME chord root+one hot chort type+one hot chord inv)
- training involes trainig MuNet(generates music with given text/rythm/chord), DeBERTA(generates beat cound and intervals from text description), FLAN-T5(generates chords for beats(from deberta) and text description)

## Fundamental Music Embeeding

https://ojs.aaai.org/index.php/AAAI/article/download/25635/25407

- space defined to reflect music domain specifis:
    - Translational Invariance: This property ensures that the relative distances between pitches (or other musical attributes) are preserved.
    - Transposability: This property allows the embedding to be shifted (transposed) in the vector space.
    - Separability: This ensures that different types of FMTs (e.g., pitch, duration) and their relative embeddings (e.g., intervals, time shifts) are distinct and orthogonal in the embedding space.
- there are separate functions for each music entity so that it is possible to differentaite between each of them

sinosuidal embedings: inspired by positional encoding in transformers, which has desirable properties like translational invariance, transposability, and continuity. The sinusoidal encoding captures the periodic nature of music and allows for these properties.

exponentialy decreasing frequency: motivated by the Fourier transform and its frequency components. The exponential function ensures that different types of FME (e.g., pitch, duration) are orthogonal and separable in the embedding space.

biases: biases serve to ensure separability between different types of music tokens

---


> MusTango, StableAudio, MuLan, CLAP

- conditioning pretrained model
    - with finetuning: Coco-Mulla, PEFT(parameter-efficient fine-tuning) - ControlNet method
    - without finetuning: usage of AudioLDM for textual inversion, SMITIN


textual inversion, double classifier, MusTango, StableAudio, Coco-mulla, AudioLDM, SMITIN, Music Foundation model, https://arxiv.org/html/2401.17800v1
Fundamental Music Embeding, onset-and-beat-based positional encoding)
