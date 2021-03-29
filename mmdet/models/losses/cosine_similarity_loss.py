import torch
import torch.nn as nn

from ..registry import LOSSES
from .utils import weighted_loss


@weighted_loss
def cosine_similarity_loss(pred, target, cos_func):
    assert isinstance(cos_func, nn.Module)
    assert pred.size() == target.size() and target.numel() > 0

    loss = cos_func(pred, target, torch.tensor(1, device=pred.device))

    return loss


@LOSSES.register_module
class CosineSimilarityLoss(nn.Module):

    def __init__(self, margin=0.0, reduction='mean', loss_weight=1.0):
        super(CosineSimilarityLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.cos = nn.CosineEmbeddingLoss(margin=margin, reduction='none')

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_cos = self.loss_weight * cosine_similarity_loss(
            pred,
            target,
            weight,
            cos_func=self.cos,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        return loss_cos
