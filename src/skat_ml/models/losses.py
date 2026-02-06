"""Loss functions for Skat ML models."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification.

    Downweights easy examples to focus training on hard/rare cases.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    When gamma=0, this is equivalent to CrossEntropyLoss.
    Higher gamma (e.g., 2) focuses more on hard examples.

    Args:
        gamma: Focusing parameter. 0 = standard CE, 2 = recommended default.
        alpha: Optional class weights tensor of shape (num_classes,).
        reduction: 'none' returns per-sample loss, 'mean' averages.
    """

    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "none"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Standard cross-entropy (per sample)
        ce_loss = F.cross_entropy(logits, targets, reduction="none")

        # Get probability of the correct class
        p_t = torch.exp(-ce_loss)

        # Focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        loss = focal_weight * ce_loss

        # Apply class weights if provided
        if self.alpha is not None:
            if self.alpha.device != targets.device:
                self.alpha = self.alpha.to(targets.device)
            alpha_t = self.alpha[targets]
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class BinaryFocalLoss(nn.Module):
    """
    Focal Loss for binary classification (AMP-safe, accepts logits).

    Downweights easy examples to focus training on hard/rare cases.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)

    When gamma=0, this is equivalent to BCEWithLogitsLoss.
    Higher gamma (e.g., 2) focuses more on hard examples.

    Args:
        gamma: Focusing parameter. 0 = standard BCE, 2 = recommended default.
        alpha: Optional weight for positive class. Use < 0.5 if positives dominate.
        reduction: 'none' returns per-sample loss, 'mean' averages.
    """

    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw logits (before sigmoid), shape (N,)
            targets: Binary targets (0 or 1), shape (N,)
        """
        # Compute BCE with logits (numerically stable, AMP-safe)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Get probabilities for focal weight calculation
        probs = torch.sigmoid(logits)

        # p_t = p if y=1, else 1-p
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Focal weight
        focal_weight = (1 - p_t) ** self.gamma

        loss = focal_weight * bce

        # Apply alpha weighting if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


def masked_bce_loss(
    logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor = None
) -> torch.Tensor:
    """
    Masked Binary Cross Entropy Loss with Logits (AMP-safe).
    Ignores targets with value -1.0.

    Args:
        logits: (batch, num_classes) raw logits (before sigmoid)
        target: (batch, num_classes) targets, -1.0 means ignored
        weight: (batch,) optional sample weights
    """
    mask = target >= 0  # True where not masked
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Select valid elements
    logits_masked = logits[mask]
    target_masked = target[mask]

    # Use binary_cross_entropy_with_logits (AMP-safe)
    loss = F.binary_cross_entropy_with_logits(logits_masked, target_masked, reduction="none")

    # Apply weights if provided
    if weight is not None:
        # weight is (batch,), mask is (batch, num_classes)
        # Broadcast weight to (batch, num_classes)
        weight_expanded = weight.unsqueeze(1).expand_as(target)
        # Select weights corresponding to valid elements
        weight_masked = weight_expanded[mask]
        loss = loss * weight_masked

    return loss.mean()
