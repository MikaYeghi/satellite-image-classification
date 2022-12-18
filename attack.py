import random
import torch
from tqdm import tqdm

from dataset import SatelliteDataset
from attacker import FGSMAttacker
from transforms import SatTransforms
from utils import load_meshes, create_model
from attacked_image import AttackedImage

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
attacker = FGSMAttacker(model, cfg.ATTACKED_PARAMS, cfg.ADVERSARIAL_SAVE_DIR)
print(attacker)

"""Loop through all possible negative samples"""
idx = 0
t = tqdm(train_set, desc=f"Index: #{idx}")
for background_image, label in tqdm(train_set):
    # Randomly select 1 mesh
    mesh = random.choice(meshes)
    
    # Initialize random initial rendering parameters
    attacked_image = AttackedImage(background_image.clone(), mesh, device=device)
        
    # Generate the adversarial example
    attacker.attack_single_image(attacked_image)
    
    t.set_description(f"Index: #{idx}. Pairs: {attacker.get_num_pairs()}")

attacker.save()