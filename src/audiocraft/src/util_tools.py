from torch.nn import functional as F
import torch
import typing as tp


def compute_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
) -> tp.Tuple[torch.Tensor, tp.List[torch.Tensor]]:
    """Compute cross entropy between multi-codebook targets and model's logits.
    The cross entropy is computed per codebook to provide codebook-level cross entropy.
    Valid timesteps for each of the codebook are pulled from the mask, where invalid
    timesteps are set to 0.

    Args:
            logits (torch.Tensor): Model's logits of shape [B, K, T, card].
            targets (torch.Tensor): Target codes, of shape [B, K, T].
            mask (torch.Tensor): Mask for valid target codes, of shape [B, K, T].
    Returns:
            ce (torch.Tensor): Cross entropy averaged over the codebooks
            ce_per_codebook (list of torch.Tensor): Cross entropy per codebook (detached).
    """
    B, K, T = targets.shape
    assert logits.shape[:-1] == targets.shape
    assert mask.shape == targets.shape
    ce = torch.zeros([], device=targets.device)
    ce_per_codebook: tp.List[torch.Tensor] = []
    for k in range(K):
        logits_k = (
            logits[:, k, ...].contiguous().view(-1, logits.size(-1))
        )  # [B x T, card]
        targets_k = targets[:, k, ...].contiguous().view(-1)  # [B x T]
        mask_k = mask[:, k, ...].contiguous().view(-1)  # [B x T]
        ce_targets = targets_k[mask_k]
        ce_logits = logits_k[mask_k]
        q_ce = F.cross_entropy(ce_logits, ce_targets)
        ce += q_ce
        ce_per_codebook.append(q_ce.detach())
    # average cross entropy across codebooks
    ce = ce / K
    return ce, ce_per_codebook


def compute_ortho_loss(emb_matrix: torch.Tensor) -> torch.Tensor:
    G = torch.matmul(emb_matrix, emb_matrix.T)
    G = G / G.norm(dim=1, keepdim=True)
    identity = torch.eye(G.size(0), device=G.device)
    off_diag = G * (1 - identity)
    return torch.norm(off_diag, p="fro") ** 2


def compute_contrastive_loss_with_labels(
    logits: torch.Tensor, labels: torch.Tensor, temperature: float = 0.5
) -> torch.Tensor:
    """
    Compute contrastive loss between logits of song_1 and song_2 using labels to determine
    whether a pair is positive or negative.

    Args:
            logits (torch.Tensor): Stacked logits for song_1 and song_2 of shape [2 * B, K, T, card].
                                                       logits[:B] are for song_1, logits[B:] are for song_2.
            labels (torch.Tensor): Binary labels (1 for positive, 0 for negative) of shape [B].
                                                       These labels determine if the pair (song_1, song_2) is a positive or negative pair.
            temperature (float): Temperature scaling for contrastive loss.

    Returns:
            contrastive_loss (torch.Tensor): The computed contrastive loss.
    """
    num_examples = (
        logits.shape[0] // 2
    )  # Assuming first half is for song_1, second half is for song_2
    assert logits.shape[0] % 2 == 0, "Logits should be stacked for song_1 and song_2."

    # Split logits for song_1 and song_2
    logits_song_1 = logits[:num_examples]  # [B, K, T, card]
    logits_song_2 = logits[num_examples:]  # [B, K, T, card]

    B, K, T, _ = logits_song_1.shape

    # Flatten the logits for each codebook (K) to create embeddings
    logits_song_1 = logits_song_1.reshape(B, -1)  # [B, K * T * card]
    logits_song_2 = logits_song_2.reshape(B, -1)  # [B, K * T * card]

    # Normalize embeddings (important for contrastive learning)
    logits_song_1 = F.normalize(logits_song_1, dim=1)
    logits_song_2 = F.normalize(logits_song_2, dim=1)

    # Compute cosine similarity between all pairs
    cosine_similarity = F.cosine_similarity(logits_song_1, logits_song_2)  # [B]

    # Contrastive loss formula (positive pairs should have higher similarity, negative lower)
    positive_loss = (1 - labels) * torch.pow(
        cosine_similarity, 2
    )  # Push negative pairs apart
    negative_loss = labels * torch.pow(
        torch.clamp(1.0 - cosine_similarity, min=0.0), 2
    )  # Bring positive pairs together

    # Average the loss over the batch
    contrastive_loss = torch.mean(positive_loss + negative_loss)

    return contrastive_loss
