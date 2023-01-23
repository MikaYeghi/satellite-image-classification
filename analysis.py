import glob
import torch
import pickle
from tqdm import tqdm
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

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
    pdb.set_trace()
                
"""Analyze the unique colors in camouflages"""
# camouflages_dir = "/home/myeghiaz/Storage/organic-camouflages"
# analyze_camouflage_unique_colors(camouflages_dir, device=device)

"""Extract the distribution of the number of vehicles in the non-centered dataset"""
annotations_dir = "/home/myeghiaz/Storage/GSD-0.125m_sample-size-50_mean-sampling-freq-1/annotations"
get_num_vehicles_distribution(annotations_dir)