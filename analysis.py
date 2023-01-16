import glob
import torch
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

# Analyze the unique colors in camouflages
camouflages_dir = "/home/myeghiaz/Storage/organic-camouflages"
analyze_camouflage_unique_colors(camouflages_dir, device=device)