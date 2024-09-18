## Datasets
- MusicCaps
- FMA
- ShutterStock
- Pond5
- MeLBench
- AudioSet Music Subset
- MagnaTagTune
- Million Song Dataset
- Free Music Archive
- Music4All
- Song Describer-Dataset

## Tasks
- long audio generation
- controllability

## Comparisons
- AudioLDM
- MusicGEN
- AudioLDM 2
- MusicGEN continuation
- MusicGEN with audio CLAP
- Riffusion
- Mousai
- MusicLM
- Noise2Music
- Jen-1
- QA-MDT

## Metrics

https://github.com/haoheliu/audioldm_eval
audioldm_eval library

### FAD

- audibility of music

### FD

- PANN's

### KL

### CLAP

### Melody accuracy(2408.04865)

Pitch matching to reference music

### Beat stability(2408.04865)

- Predominant Local Pulse estimation

### Nearest Neighbours in Common(2407.12563)

- number of closest one songs in dataset between conditioning audio and generated audio
- embedings for audio chunks -> cos sim between conditioning audio -> find closes ones

### G is the Nearest Neighbor of C(2407.12563)

- 1 if generated audio is NN of conditioning audio

### Human asked
- How would you rate the overall quality of this excerpt?
- Without considering audio quality, how similar are these two excerpts in terms of style?
- Without considering audio quality, how likely do you think these two excerpts are from the same song?
