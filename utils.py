import os
import glob
import math
import shutil
from torch import nn
from matplotlib import pyplot as plt
import torch
from random import shuffle, uniform
import torchvision.models as models
from tqdm import tqdm
from pytorch3d.io import load_objs_as_meshes
from sklearn.model_selection import train_test_split
from pathlib import Path

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

def initialize_empty_model(cfg, device='cuda'):
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
    return model

def create_model(cfg, device='cuda'):
    model = initialize_empty_model(cfg, device)
    
    # Check the number of GPU-s
    if cfg.NUM_GPUS > 1:
        model = nn.DataParallel(model)
        
        # Load model weights
        if cfg.MODEL_WEIGHTS:
            print(f"Loading model weights from {cfg.MODEL_WEIGHTS}")
            model.module.load_state_dict(torch.load(cfg.MODEL_WEIGHTS))
    else:
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

def generate_train_test(dataset_dir, save_dir, split_ratio=0.8):
    """
    This function takes a non-split dataset and randomly splits it into training and testing sets. 
    Split ratio shows which portion of the initial dataset is the training set.
    Raw dataset should have the following structure:
    
    dataset_dir/
    ├── negative
    └── positive
    
    Where the "negative" and "positive" directories contain empty and non-empty images respectively (non-empty meaning there is a
    vehicle in the image).
    
    The generated dataset is stored with the following structure:
    
    save_dir/
    ├── test
    │   ├── negative
    │   └── positive
    └── train
        ├── negative
        └── positive
    """
    image_formats = ['.jpg', '.png']
    positive_images_dir = os.path.join(dataset_dir, "positive")
    negative_images_dir = os.path.join(dataset_dir, "negative")
    
    # Extract image paths
    positive_images_paths = []
    negative_images_paths = []
    for image_format in image_formats:
        positive_images = glob.glob(positive_images_dir + "/*" + image_format)
        negative_images = glob.glob(negative_images_dir + "/*" + image_format)
        
        positive_images_paths = positive_images_paths + positive_images
        negative_images_paths = negative_images_paths + negative_images
    
    print(f"Extracted {len(positive_images_paths)} positive images and {len(negative_images_paths)} negative images.")
    
    # Shuffle and randomly split the positive and negative lists
    shuffle(positive_images_paths)
    shuffle(negative_images_paths)
    
    if split_ratio > 0 and split_ratio < 1:
        train_pos, test_pos = train_test_split(positive_images_paths, train_size=split_ratio)
        train_neg, test_neg = train_test_split(negative_images_paths, train_size=split_ratio)
    elif split_ratio == 0:
        train_pos = []
        train_neg = []
        test_pos = positive_images_paths.copy()
        test_neg = negative_images_paths.copy()
    elif split_ratio == 1:
        train_pos = positive_images_paths.copy()
        train_neg = negative_images_paths.copy()
        test_pos = []
        test_neg = []
    else:
        raise ValueError('Split ratio must be in the range [0, 1]!')
    
    # Create the save directories
    train_pos_dir = os.path.join(save_dir, "train", "positive")
    train_neg_dir = os.path.join(save_dir, "train", "negative")
    test_pos_dir = os.path.join(save_dir, "test", "positive")
    test_neg_dir = os.path.join(save_dir, "test", "negative")
    Path(train_pos_dir).mkdir(parents=True, exist_ok=True)
    Path(train_neg_dir).mkdir(parents=True, exist_ok=True)
    Path(test_pos_dir).mkdir(parents=True, exist_ok=True)
    Path(test_neg_dir).mkdir(parents=True, exist_ok=True)
    
    # Copy the files
    print("Copying positive train files...")
    for train_pos_ in train_pos:
        shutil.copy(train_pos_, train_pos_dir)
    print("Copying negative train files...")
    for train_neg_ in train_neg:
        shutil.copy(train_neg_, train_neg_dir)
    print("Copying positive test files...")
    for test_pos_ in test_pos:
        shutil.copy(test_pos_, test_pos_dir)
    print("Copying negative test files...")
    for test_neg_ in test_neg:
        shutil.copy(test_neg_, test_neg_dir)
    
    print("Completed dataset generation!")
    
def blend_images(A, B, alpha_A=0.5, alpha_B=0.5):
    """
    Blending 2 images following https://en.wikipedia.org/wiki/Alpha_compositing.
    """
    assert alpha_A + alpha_B == 1, "Alpha-s must sum to 1!" # QUESTIONABLE!!!
    alpha_O = alpha_A + alpha_B * (1 - alpha_A)
    C = (alpha_A * A + alpha_B * (1 - alpha_A) * B) / alpha_O
    return C

def save_checkpoint(cfg, model, epoch, iter_counter, is_final):
    if is_final:
        save_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pt")
    else:
        save_path = os.path.join(cfg.OUTPUT_DIR, f"model_{iter_counter}.pt")
    
    print(f"Saving checkpoint to {save_path}")
    
    model_state_dict = model.state_dict() if cfg.NUM_GPUS == 1 else model.module.state_dict()
    torch.save({
        "epoch": epoch,
        "iter_counter": iter_counter,
        "model_state_dict": model_state_dict,
    }, save_path)

def load_checkpoint(cfg, device='cuda'):
    if cfg.MODEL_WEIGHTS:
        print(f"Loading checkpoint from {cfg.MODEL_WEIGHTS}")
        # If there is a model checkpoint, load it
        if cfg.MODEL_WEIGHTS.split('.')[-1] == 'pth':
            # Load as the old checkpoint
            epoch = 0
            iter_counter = 0
            model = create_model(cfg, device)
        elif cfg.MODEL_WEIGHTS.split('.')[-1] == 'pt':
            # Load as the new checkpoint
            checkpoint = torch.load(cfg.MODEL_WEIGHTS)
            
            # Load side parameters
            epoch = checkpoint['epoch'] + 1
            iter_counter = checkpoint['iter_counter']
            
            # Load model weights
            model = initialize_empty_model(cfg, device)
            if cfg.NUM_GPUS > 1:
                model = nn.DataParallel(model)

                # Load model weights
                if cfg.MODEL_WEIGHTS:
                    model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Load model weights
                if cfg.MODEL_WEIGHTS:
                    model.load_state_dict(checkpoint['model_state_dict'])
        else:
            raise NotImplementedError
    else:
        # If there is no model checkpoint, simply initialize the epoch, iteration counter and model
        epoch = 0
        iter_counter = 0
        model = initialize_empty_model(cfg, device)
    
    model.to(device)
    return (epoch, iter_counter, model)