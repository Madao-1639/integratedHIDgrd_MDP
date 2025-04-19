import torch
from torch import nn
import torch.nn.functional as F

def _loss_reduction(loss,reduction: str):
    if reduction == "none":
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(
            f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
        )

def FocalLoss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    ce_loss = F.binary_cross_entropy(y_pred, y_true, reduction="none")
    p_t = y_pred * y_true + (1 - y_pred) * (1 - y_true)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if 0 < alpha < 1:
        alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
        loss = alpha_t * loss
    return _loss_reduction(loss,reduction)

def MVFLoss(hi_f,m=1,reduction="none",):
    loss = torch.square(hi_f-m)
    return _loss_reduction(loss,reduction)

def MONLoss(hi_pre,hi_cur,c=0,penalty_type='exp',reduction="none",):
    diff = hi_pre - hi_cur + c
    if penalty_type == 'linear':
        inner_term = diff
    elif penalty_type == 'exp':
        inner_term = torch.exp(diff)-1
    elif penalty_type == 'tanh':
        inner_term = torch.tanh(diff)
    else:
        raise ValueError(
            f"Invalid Value for arg 'penalty_type': '{penalty_type} \n Supported reduction modes: 'linear', 'exp', 'tanh'"
        )
    loss = F.relu(inner_term)
    return _loss_reduction(loss,reduction)

def CONLoss(hi_ppre,hi_pre,hi_cur,**mon_kwargs):
    d_pre = hi_ppre - hi_pre
    d_cur = hi_pre - hi_pre
    return MONLoss(d_pre,d_cur,**mon_kwargs)
