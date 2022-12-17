import torch
from torchvision import transforms
from matplotlib import pyplot as plt

from dataset import SatelliteDataset
from utils import extract_samples
from satadv import SatAdv
from transforms import SatTransforms

import config as cfg
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

import pdb

"""Load the dataset"""
transforms = SatTransforms()
train_transform = transforms.get_train_transforms()
test_transform = transforms.get_test_transforms()
train_set = SatelliteDataset(cfg.TRAIN_PATH, transform=train_transform, shuffle=True, device=device)
test_set = SatelliteDataset(cfg.TEST_PATH, transform=test_transform, shuffle=True, device=device)

"""Extract samples of background"""
samples = extract_samples(test_set, 1, 3)

"""Plot the samples"""
# k = 0
# for sample in samples:
#     img = sample[0].cpu().clone().permute(1,2,0).numpy()
#     plt.imshow(img)
#     plt.savefig(f"results/img_{k}.jpg")
#     plt.close('all')
#     print(f"Sample {k}, label {sample[1]}")
#     k += 1

"""Initialize the adversarial attacker"""
adv_net = SatAdv(cfg)

"""Print model details"""
# for name, param in adv_net.named_parameters():
#     if param.requires_grad:
#         print(name)

"""Generate a sample synthetic image"""
mesh = adv_net.meshes[0].clone()
background_image = samples[0][0].clone()
# sample_img = adv_net.render_synthetic_image(mesh, background_image)

"""Attack an image"""
# adv_net.attack_image_mesh(mesh, background_image)
# adv_net.find_failure_regions(mesh, background_image, resolution=50)
adv_net.failure_analysis(test_set, resolution=25, n_samples=100, intensity=1.0)

"""Generate synthetic dataset"""
# adv_net.generate_synthetic_dataset(train_set, test_set)