from torch.nn import functional as F
import torch
import typing as tp
def compute_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor
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
		logits_k = logits[:, k, ...].contiguous().view(-1, logits.size(-1))  # [B x T, card]
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