import random
import torch
from tqdm import tqdm
from pytorch3d.renderer import TexturesUV

from dataset import SatelliteDataset
from attacker import FGSMAttacker, UnifiedTexturesAttacker
from transforms import SatTransforms
from utils import load_meshes, create_model, load_checkpoint
from attacked_image import AttackedImage

import config as cfg
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

import pdb

"""Load the data set and remove all positive samples"""
transforms = SatTransforms()
train_transform = transforms.get_train_transforms()
test_transform = transforms.get_test_transforms()
train_set = SatelliteDataset(cfg.TRAIN_PATH, transform=train_transform, device=device)
test_set = SatelliteDataset(cfg.TEST_PATH, transform=test_transform, device=device)
# train_set.remove_positives()

"""K-means analysis"""
# train_set.remove_positives()
# train_set.leave_fraction_of_negatives(0.05)
# test_set.leave_fraction_of_negatives(0.05)
# train_set.KMeansAnalysis(K_max=10)
# exit()

"""Load the meshes"""
# meshes = load_meshes(cfg, shuffle_=True, device='cpu')

"""Initialize the model"""
# model = create_model(cfg, device)
_, _, model = load_checkpoint(cfg, device)

"""Initialize the attacker"""
attacker = UnifiedTexturesAttacker(model, train_set, test_set, cfg, device=device)
print(attacker)

"""Perform the attack"""
# adversarial_texture_map = attacker.attack()
attacker.evaluate()
exit()

"""Loop through all possible negative samples"""
idx = 0
train_set.leave_number_of_negatives(cfg.NUM_ADV_IMGS)
t = tqdm(train_set, desc="Pairs: 0")
for background_image, label in tqdm(train_set):
    # Randomly select 1 mesh
    mesh = random.choice(meshes).clone().to(device)
    
    # Initialize random initial rendering parameters
    attacked_image = AttackedImage(background_image.clone(), mesh, device=device)
        
    # Generate the adversarial example
    # attacker.attack_single_image(attacked_image)
    attacker.EOT_attack_scene(attacked_image)
    
    t.set_description(f"Pairs: {attacker.get_num_pairs()}")
    
    # if attacker.get_num_pairs() >= cfg.NUM_ADV_IMGS:
    #     break
    idx += 1

attacker.save()