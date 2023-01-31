import os
import glob
import torch
import pickle
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from lib.gmm.gmm import GaussianMixture
from torchvision.transforms.functional import pil_to_tensor

from dataset import SatelliteDataset
from transforms import SatTransforms
from utils import load_checkpoint

import config as cfg
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

import pdb

def analyze_camouflage_unique_colors(camouflages_dir, device='cuda'):
    """
    This function finds the unique colors in the camouflages.
    """
    camouflages = {}
    camouflage_paths = glob.glob(camouflages_dir + "/*.png")
    
    # Load the camouflages as pytorch tensors
    for camouflage_path in camouflage_paths:
        camouflage_name = camouflage_path.split('/')[-1]
        camouflage = Image.open(camouflage_path).convert("RGB")
        camouflages[camouflage_name] = (pil_to_tensor(camouflage).permute(1, 2, 0).to(device))
    print(camouflages)
    # Extract unique colors for each camouflage
    unique_colors = {}
    for camouflage in camouflages:
        image = camouflages[camouflage]
        rows, cols, _ = image.shape
        colors = set()
        for i in range(rows):
            for j in range(cols):
                colors.add(image[i][j])    

def get_num_vehicles_distribution(annotations_dir):
    total_num_vehicles = 0
    num_vehicles_distribution = {}
    annotation_files = glob.glob(annotations_dir + "/*.pkl")
    for annotation_file in tqdm(annotation_files):
        with open(annotation_file, 'rb') as f:
            data = pickle.load(f)
        num_vehicles = len(data['object_locations']['small'][0])
        if num_vehicles == 0:
            pass
        else:
            if num_vehicles in num_vehicles_distribution.keys():
                num_vehicles_distribution[num_vehicles] += 1
            else:
                num_vehicles_distribution[num_vehicles] = 1
            total_num_vehicles += 1
    text = ""
    for num_vehicles in num_vehicles_distribution:
        # num_vehicles_distribution[num_vehicles] /= total_num_vehicles
        text += f"{num_vehicles} vehicles: {num_vehicles_distribution[num_vehicles]}\n"
    text += f"Total: {total_num_vehicles}"
    print(text)
    
def pixels_EM_analysis(data_path):
    pixels = torch.empty(size=(0, 3), device='cuda')
    
    # Initialize the dataset
    transforms = SatTransforms()
    test_transform = transforms.get_test_transforms()
    train_set = SatelliteDataset([data_path], transform=test_transform, device='cuda')
    train_set.remove_positives()
    train_set.leave_fraction_of_negatives(0.1)
    print(train_set.details())
    
    # Extract pixel values
    print("Extracting pixel values...")
    for image, _ in tqdm(train_set):
        image = image.permute(1, 2, 0)
        
        # Get the pixel values
        pixels_ = image.view(-1, 3)
        pixels = torch.cat((pixels, pixels_))
    
    # Initialize the GMM model
    n_components = 5
    n_features = 3
    model = GaussianMixture(n_components, n_features)
    
    # Fit the data
    print("Fitting the GMM model to the data...")
    model = model.cuda()
    pixels = pixels.cuda()
    model.fit(pixels)
    
    # Extract the results
    mu = model.mu.squeeze()
    var = model.var.squeeze()
    pi = model.pi.squeeze()
    
    # Save the results
    print("Saving the results...")
    torch.save(mu, "results/mus.pt")
    torch.save(var, "results/vars.pt")
    torch.save(pi, "results/pis.pt")
    
def analyze_quadrant_performance(cfg, quadrant_size, anns_dir, data_dir, device):
    # Initialize the heatmap of correct probabilities
    assert 50 % quadrant_size == 0, "Image size (50x50) must be divisible by the quadrant size!"
    heatmap_size = 50 // quadrant_size
    pred_quadrants = torch.zeros(size=(heatmap_size, heatmap_size))
    gt_quadrants = torch.zeros(size=(heatmap_size, heatmap_size))
    
    # Load the model and the transform
    _, _, model = load_checkpoint(cfg, device)
    transforms = SatTransforms()
    transform = transforms.get_test_transforms()
    activation = nn.Sigmoid()
    
    # Extract files with single vehicles only and run through the model
    single_vehicle_files = []
    positive_images_list = glob.glob(data_dir + "/*/positive/*.jpg")
    for image_path in tqdm(positive_images_list):
        encoding = image_path.split('/')[-1][:-4]
        anns_path = os.path.join(anns_dir, f"{encoding}.pkl")
        
        # Load the annotations
        with open(anns_path, 'rb') as f:
            data = pickle.load(f)
            anns = data['object_locations']['small'][0]
            num_vehicles = len(anns)
        
        if num_vehicles == 1:
            # Run the image through the model
            image = transform(Image.open(image_path).convert('RGB')).unsqueeze(0).to(device)
            pred = 1 - activation(model(image)) # probability of the correct class
            
            # Extract the quadrant coordinates
            quadrant_coords = (anns // quadrant_size).astype('int').squeeze().tolist()
            quadrant_x = quadrant_coords[0]
            quadrant_y = quadrant_coords[1]
            
            # Populate the GT heatmap
            gt_quadrants[quadrant_x][quadrant_y] += 1.0
            
            # Populate the prediction heatmap
            pred_quadrants[quadrant_x][quadrant_y] += pred.item()
        
    # Generate and save the correctness heatmap
    heatmap = torch.div(pred_quadrants, gt_quadrants)
    plt.imshow(heatmap, cmap='Greys')
    plt.savefig("results/non-centered-accuracy-heatmap.png", dpi=100)
    plt.close()
    
    # Generate and save the GT quadrants heatmap
    plt.imshow(gt_quadrants, cmap='Greys')
    plt.savefig("results/non-centered-GT-heatmap.png", dpi=100)
    plt.close()

"""Analyze the unique colors in camouflages"""
# camouflages_dir = "/home/myeghiaz/Storage/organic-camouflages"
# analyze_camouflage_unique_colors(camouflages_dir, device=device)

"""Extract the distribution of the number of vehicles in the non-centered dataset"""
# annotations_dir = "/home/myeghiaz/Storage/GSD-0.125m_sample-size-50_mean-sampling-freq-1/annotations"
# get_num_vehicles_distribution(annotations_dir)

"""Perform EM on data pixels"""
# data_path = "/home/myeghiaz/Storage/SatClass-Real-0.125m-50px/train"
# pixels_EM_analysis(data_path)

"""Analyze model performance on quadrants in non-centered data"""
quadrant_size = 5
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
anns_dir = "/home/myeghiaz/Storage/GSD-0.125m_sample-size-50_mean-sampling-freq-1/annotations"
data_dir = "/home/myeghiaz/Storage/SatClass-Real-non-centered-0.125m-50px-no-margin"
analyze_quadrant_performance(cfg, quadrant_size, anns_dir, data_dir, device)