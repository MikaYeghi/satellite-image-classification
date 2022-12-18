import os
import glob
import math
from torch import nn
from matplotlib import pyplot as plt
import torch
from random import shuffle, uniform
import torchvision.models as models
from tqdm import tqdm
from pytorch3d.io import load_objs_as_meshes

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
        azimuth = math.atan(y / x) * 180.0 / math.pi
        if x < 0:
            if y > 0:
                azimuth += 180
            else:
                azimuth -= 180
    
    return (elevation, azimuth)

def create_model(cfg, device='cuda'):
    # Get the model name
    model_name = cfg.MODEL_NAME
    
    # Initialize the model
    if model_name == 'resnet101':
        print("Initializing a ResNet-101 model.")
        model = models.resnet101(pretrained=True)
        model.fc = torch.nn.Linear(2048, 1, device=device, dtype=torch.float32)
    elif model_name == 'vgg16':
        print("Initializing a VGG-16 model.")
        model = models.vgg16(pretrained=True)
        model.classifier[6] = torch.nn.Linear(4096, 1, device=device, dtype=torch.float32)
    else:
        raise NotImplementedError
    
    # Load model weights
    if cfg.MODEL_WEIGHTS:
        print(f"Loading model weights from {cfg.MODEL_WEIGHTS}")
        model.load_state_dict(torch.load(cfg.MODEL_WEIGHTS))
    
    # Move the model to the device
    model.to(device)
    
    return model
    
def load_meshes(cfg, shuffle_=True, device='cuda'):
    print(f"Loading meshes from {cfg.MESHES_DIR}")
    meshes = []
    obj_paths = glob.glob(cfg.MESHES_DIR + "/*.obj")
    for obj_path in tqdm(obj_paths):
        mesh = load_objs_as_meshes([obj_path], device=device)[0]
        meshes.append(mesh)
    if shuffle_:
        shuffle(meshes)
    return meshes    

def get_lightdir_from_elaz(elev, azim, device='cuda'):
    x = -math.cos(math.radians(elev)) * math.sin(math.radians(azim))
    y = -math.sin(math.radians(elev))
    z = -math.cos(math.radians(elev)) * math.cos(math.radians(azim))
    xyz = torch.tensor([x, y, z], device=device).unsqueeze(0)
    return xyz