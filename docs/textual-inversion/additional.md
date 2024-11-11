LoRA: Low-Rank Adaptation of Large Language Models
https://openreview.net/forum?id=nZeVKeeFYf9

However, due to the self-supervised nature within the limited latent space
of the pre-trained diffusion model, the vanilla textual inversion often results in varied performance
in terms of quality and efficiency for different image sets, requiring meticulous adjustments for
learning rates and iteration counts.

https://arxiv.org/pdf/2408.04785v1

Many concepts are better described with multiple words
than just one; "orange cat" is more descriptive than "cat". We add an auxiliary pseudoword "<fkf>", which
we refer to as the "bonus" token, and corresponding embedding wembed the exact same information. We want to encourage the two embeddings to be orthogonal,

We tested a few token strategies for each model:
1. Default: identical to Gal et al. (2022).
2. Multi 10: based off of ProSpect (Zhang et al., 2023b), we have ten tokens, each corresponding to
five inference steps
3. Multi 50: based off of ProSpect (Zhang et al., 2023b), we have a separate token for each inference
step, for a total of 50 tokens.
4. Negative: based off DreamArtist (Dong et al., 2023), we use a negative token p−, and the loss is
||ϵ − f(ϵθ(zt, t, cϕ(p)), ϵθ(zt, t, cϕ(p−))), where f(a, b) = b + γ(a − b).
5. Bonus: using a bonus token in addition to the placeholder token, with the orthogonal loss between
the original placeholder token and the bonus
6. Triple Spare: using three bonus tokens in addition to the placeholder token, with the orthogonal
loss between all combinations of the placeholder and bonus tokens (i.e. for one placeholder token
and three spares, we would have 12 different orthogonal loss terms)

https://arxiv.org/pdf/2405.20607v2
Subsequently, self-supervised refinement refines these pseudo words
through contrastive loss computation between images and texts, enhancing the fidelity of generated reports to images

Through textual inversion, the pseudo words obtained by transforming image
embeddings contain both image features and linguistic spatial characteristics.
Textual inversion can eliminate the spatial gap effectively, making the features of two modalities be computed in the common compact space.

We then perform self-supervised refinement by calculating contrastive loss between the obtained pseudo words and image features.