import os
import math
from torch import nn
from matplotlib import pyplot as plt
import torch
from random import shuffle, uniform

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

def random_unique_split(original_list, len1, len2):
    assert len1 + len2 == len(original_list), "The sum of sublist lengths is different from the original list length!"
    shuffle(original_list)
    sublist1 = original_list[:len1]
    sublist2 = original_list[len1:]
    
    return (sublist1, sublist2)

def sample_random_elev_azimuth(x_min, y_min, x_max, y_max, distance):
    """
    This function samples x and y coordinates on a plane, and converts them to elevation and azimuth angles.
    There is a trick when generating the azimuth angle: the resulting angle is doubled, since atan covers only (-pi, pi) range. Thus, to cover the full range, the angle is doubled.
    
    It was found that x_min = y_min = -1.287 and x_max = y_max = 1.287 result in the best angles, where elevation ranges roughly from 70 to 90, and azimuth goes from 0 to 360.
    """
    x = uniform(x_min, x_max)
    y = uniform(y_min, y_max)
    
    if x == 0 and y == 0:
        elevation = 90.0
        azimuth = 0.0
    elif x == 0:
        elevation = math.atan(distance / math.sqrt(x * x + y * y)) * 180.0 / math.pi
        azimuth = 0.0
    else:
        elevation = math.atan(distance / math.sqrt(x * x + y * y)) * 180.0 / math.pi
        azimuth = math.atan(y / x) * 180.0 / math.pi * 2.0 # added a factor of 2 to cover the entire 360 range
    
    return (elevation, azimuth)