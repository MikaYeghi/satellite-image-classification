import os
from torch import nn
from matplotlib import pyplot as plt
import torch

import pdb

def make_train_step(model, loss_fn, optimizer):
    def train_step(images_batch, labels_batch):
        activation = nn.Sigmoid()
        model.train()
        yhat = model(images_batch)
        yhat = activation(yhat)
        loss = loss_fn(yhat, labels_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    return train_step

def plot_training_info(train_losses, val_losses, save_dir='results'):
    plt.figure()
    plt.subplot(211)
    plt.plot(train_losses, 'b')
    plt.grid(True)
    plt.xlabel("Iteration number")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Training loss")
    plt.subplot(212)
    plt.plot(val_losses, 'b')
    plt.grid(True)
    plt.xlabel("Iteration number")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Validation loss")
    
    save_dir = os.path.join(save_dir, "results.jpg")
    print(f"Saving the graphs to {save_dir}")
    plt.savefig(save_dir)
    
def get_F1_stats(preds, targets):
    # Convert 0->1 and 1->0 for ease
    # Now 1 means vehicles and 0 background
    preds = 1 - preds
    targets = 1 - targets
    assert len(preds) == len(targets)
    TP = int(torch.sum(targets * preds).item())
    FP = int(torch.sum((1 - targets) * preds).item())
    FN = int(torch.sum(targets * (1 - preds)).item())
    return (TP, FP, FN)

def extract_samples(dataset, category_id, n_samples):
    samples = []
    for image, label in dataset:
        if len(samples) == n_samples:
            break
        if label == category_id:
            samples.append((image, label))
    return samples