from torch.nn import functional as F
import torch
import typing as tp
from audiocraft.models.lm import LMOutput


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


def compute_ce_by_concept(concepts: list[str], music: torch.Tensor, out: LMOutput):
    u_concepts = set(concepts)
    res = 0
    for concept in u_concepts:
        mask = torch.tensor([c == concept for c in concepts], dtype=torch.bool)
        ce_loss, _ = compute_cross_entropy(
            out.logits[mask], music[mask], out.mask[mask]
        )
        res += ce_loss
    return res
