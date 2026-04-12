"""
ProgLoss + STAL Loss Module
Improved loss functions for better small object detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from ..core import register


@register()
class ProgLoss(nn.Module):
    """Probabilistic Loss for improved small object detection.

    This loss function uses a probabilistic approach to handle class imbalance
    and focuses on hard-to-classify examples, particularly beneficial for
    small objects in IoT, robotics, and aerial imagery applications.

    Args:
        alpha (float): Balancing factor for positive/negative samples. Default: 0.25
        gamma (float): Focusing parameter for hard examples. Default: 2.0
        num_classes (int): Number of object categories. Default: 80
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, num_classes: int = 80):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, pred_logits: torch.Tensor, target_classes: torch.Tensor, 
                pred_boxes: torch.Tensor = None, target_boxes: torch.Tensor = None,
                ious: torch.Tensor = None) -> torch.Tensor:
        """Compute probabilistic loss.

        Args:
            pred_logits: Predicted logits (batch, num_queries, num_classes)
            target_classes: One-hot encoded target classes
            pred_boxes: Predicted boxes (for IoU weighting, optional)
            target_boxes: Target boxes (for IoU weighting, optional)
            ious: Pre-computed IoUs for weighting

        Returns:
            Loss value
        """
        pred_probs = pred_logits.sigmoid().detach()
        target_score = target_classes.float()

        if ious is not None and pred_boxes is not None and target_boxes is not None:
            iou_weight = ious.unsqueeze(-1).clamp(0, 1)
            target_score = target_score * iou_weight

        weight = self.alpha * pred_probs.pow(self.gamma) * (1 - target_classes) + target_score

        loss = F.binary_cross_entropy_with_logits(
            pred_logits, target_score, weight=weight, reduction='none'
        )
        return loss.mean(1).sum() * pred_logits.shape[1] / max(target_score.sum(), 1)


@register()
class STALLoss(nn.Module):
    """Self-Training Augmentation Loss for improved small object detection.

    STAL encourages consistency between predictions on original and augmented
    images, helping the model learn more robust features especially for small objects.

    Args:
        T (float): Temperature for softening probability distributions. Default: 2.0
        alpha (float): Weight for the distillation loss. Default: 0.5
    """

    def __init__(self, T: float = 2.0, alpha: float = 0.5):
        super().__init__()
        self.T = T
        self.alpha = alpha

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                student_boxes: torch.Tensor = None, teacher_boxes: torch.Tensor = None,
                fg_mask: torch.Tensor = None) -> torch.Tensor:
        """Compute STAL loss.

        Args:
            student_logits: Student model predictions
            teacher_logits: Teacher (EMA) model predictions
            student_boxes: Student box predictions
            teacher_boxes: Teacher box predictions
            fg_mask: Foreground mask for valid predictions

        Returns:
            STAL loss value
        """
        if teacher_logits is None:
            return torch.tensor(0.0, device=student_logits.device)

        student_probs = F.log_softmax(student_logits / self.T, dim=-1)
        teacher_probs = F.softmax(teacher_logits.detach() / self.T, dim=-1)

        distillation_loss = F.kl_div(
            student_probs, teacher_probs, reduction='none'
        ).sum(-1)

        if fg_mask is not None:
            distillation_loss = distillation_loss[fg_mask]

        loss = distillation_loss.mean() * (self.T ** 2)
        return self.alpha * loss


@register()
class SmallObjectLoss(nn.Module):
    """Specialized loss component for small object detection enhancement.

    Applies additional focus on small objects by using higher weights for
    predictions with smaller bounding boxes.

    Args:
        size_threshold (float): Threshold for considering an object as small. Default: 32*32
        weight_factor (float): Additional weight multiplier for small objects. Default: 2.0
    """

    def __init__(self, size_threshold: float = 32 * 32, weight_factor: float = 2.0):
        super().__init__()
        self.size_threshold = size_threshold
        self.weight_factor = weight_factor

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor,
                base_loss: torch.Tensor, fg_mask: torch.Tensor) -> torch.Tensor:
        """Compute small object enhanced loss.

        Args:
            pred_boxes: Predicted boxes (cxcywh format)
            target_boxes: Target boxes (cxcywh format)
            base_loss: Base loss value to be weighted
            fg_mask: Foreground mask

        Returns:
            Weighted loss with small object focus
        """
        if pred_boxes is None or target_boxes is None or fg_mask is None:
            return base_loss

        box_sizes = (target_boxes[..., 2] * target_boxes[..., 3]).clamp(min=1)
        small_object_mask = box_sizes < self.size_threshold

        if fg_mask.any():
            combined_mask = fg_mask & small_object_mask
            if combined_mask.any():
                loss_weight = torch.ones_like(base_loss)
                loss_weight[combined_mask] = self.weight_factor
                return base_loss * loss_weight.mean()

        return base_loss
