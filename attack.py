import torch

from dataset import SatelliteDataset
from attacker import FGSMAttacker
from transforms import SatTransforms
from utils import load_meshes, generate_render_params, create_model

import config as cfg
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

import pdb

"""Load the data set and remove all positive samples"""
transforms = SatTransforms()
train_transform = transforms.get_train_transforms()
train_set = SatelliteDataset(cfg.TRAIN_PATH, transform=train_transform, device=device)
train_set.remove_positives()

"""Load the meshes"""
meshes = load_meshes(cfg, shuffle_=True, device=device)

"""Initialize the model"""
model = create_model(cfg, device)

"""Initialize the attacker"""
attacker = FGSMAttacker(model, cfg.ATTACKED_PARAMS)
print(attacker)

"""Loop through all possible negative samples"""
for background_image, label in train_set:
    for mesh in meshes:
        # Initialize random initial rendering parameters
        rendering_params = generate_render_params(background_image, mesh)
        
        # Generate the adversarial example
        attacker.attack_single_image(rendering_params=rendering_params)
        break
    break