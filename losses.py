import torch
import torch.nn.functional as F

import config as cfg

def BCELoss():
    return torch.nn.BCELoss()

def FocalLoss(
    reduction='none',
    alpha: float = cfg.FOCAL_LOSS['alpha'],
    gamma: float = cfg.FOCAL_LOSS['gamma']
    ):
    def FocalLoss_(
        inputs: torch.Tensor,
        targets: torch.Tensor
        ):
        ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss
        
        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        else:
            loss = loss.sum()
        
        return loss

    return FocalLoss_

def ColorForce(color=(1.0, 1.0, 0.4), device='cuda'):
    def ColorForce_(image):
        mean = -torch.abs(torch.mean(image - color))
        std = -torch.std(image - color)
        print(f"Mean: {mean.item()}, std: {std.item()}")
        return mean + std
    color = torch.tensor(color, device=device).unsqueeze(0)
    return ColorForce_

def BCEColor(lambda_label, lambda_color, color=(1.0, 1.0, 0.4), device='cuda'):
    def ColorForce_(preds, targets, image):
        # BCE loss term
        bce_loss = bce_loss_fn(preds, targets)
        
        # Color term
        mean = -torch.abs(torch.mean(image - color))
        std = -torch.std(image - color)
        color_loss = mean + std
        
        print(f"Mean: {mean.item()}, std: {std.item()}, BCE: {bce_loss.item()}")
        return lambda_label * bce_loss + lambda_color * color_loss
    color = torch.tensor(color, device=device).unsqueeze(0)
    bce_loss_fn = torch.nn.BCELoss()
    return ColorForce_