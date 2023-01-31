import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import multivariate_normal

import config as cfg

import pdb

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

class ClassificationScore(nn.Module):
    """
    This loss function returns the probability of the correct class for each prediction.
    In other words, it penalizes high values of correct predictions. Used for adversarial attacks.
    
    inputs:
        - predictions: predictions of each class (NOTE: currently realized for 1 class classification only)
    outputs:
        - a set of probabilities of the correct classes
    """
    def __init__(self, class_id, coefficient=1.0):
        super(ClassificationScore, self).__init__()
        
        self.class_id = class_id
        self.coefficient = coefficient
    
    def forward(self, predictions, adv_patch):
        loss = torch.mean(self.class_id * predictions + (1 - self.class_id) * (1 - predictions))
        return self.coefficient * loss

class TVCalculator(nn.Module):
    """
    Module providing the functionality necessary to calculate the total variation (TV) of an adversarial patch.
    
    inputs:
        - adv_patch: adversarial patch of shape (B, C, H, W), where B is the batch size, C is the number of channels,
        H and W are the height and width of the patch respectively.
    """
    def __init__(self, coefficient=1.0):
        super(TVCalculator, self).__init__()
        
        self.coefficient = coefficient
        
    def forward(self, predictions, adv_patch):
        tv = 0
        for i in range(adv_patch.size(0)):
            tvcomp1 = torch.norm(adv_patch[i, :, :, 1:] - adv_patch[i, :, :, :-1], p=2)
            tvcomp2 = torch.norm(adv_patch[i, :, 1:, :] - adv_patch[i, :, :-1, :], p=2)
 
            tv += tvcomp1 + tvcomp2
        return self.coefficient * tv / torch.numel(adv_patch)
    
class GMMLoss(nn.Module):
    """
    Module to calculate the GMM loss, which is the negative log likelihood of the GMM pdf.
    """
    def __init__(self, mus, variances, pis, dimensions=3, coefficient=1.0, device='cuda'):
        super(GMMLoss, self).__init__()
        assert len(mus) == len(variances) == len(pis), "Number of mu-s, variances and pi-s must match."
        self.n_components = len(mus)
        self.coefficient = coefficient
        self.mus = mus
        self.variances = variances
        self.pis = pis
        self.dimensions = dimensions
        self.device = device
        
    def forward(self, predictions, adv_patch):
        loss = 0
        for i in range(self.n_components):
            # Obtain distribution parameters
            mu = self.mus[i].to(self.device)
            variance = self.variances[i].to(self.device)
            pi = self.pis[i].to(self.device)
            
            # Form the input vector
            x = adv_patch.view(-1, 3)
            
            # Compute the value of the gaussian
            loss += pi * torch.mean(self.gaussian_function(x, mu, variance))
        loss = -torch.log(loss)
        return self.coefficient * loss
    
    def gaussian_function(self, x, mu, variance):
        dist = multivariate_normal.MultivariateNormal(loc=mu, covariance_matrix=variance)
        probability = torch.exp(dist.log_prob(x))
        return probability
    
class NonPrintabilityScore(nn.Module):
    """
    Non-printability score as defined in "Physical Adversarial Attacks on an Aerial Imagery Object Detector".
    """
    def __init__(self, printable_colors, coefficient=1.0, device='cuda'):
        super(NonPrintabilityScore, self).__init__()
        self.printable_colors = printable_colors.to(device)
        self.coefficient = coefficient
        self.device = device
        
    def forward(self, predictions, adv_patch):
        nps = 0
        adv_patch_pixels = adv_patch.view(-1, 3) # Flatten the adversarial patch
        distances_matrix = torch.cdist(adv_patch_pixels, self.printable_colors)
        closest_distances = torch.min(distances_matrix, dim=1)[0]
        nps = torch.mean(closest_distances)
        return self.coefficient * nps