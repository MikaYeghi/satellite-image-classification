import torch

def BCELoss():
    return torch.nn.BCELoss()

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