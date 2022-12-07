import torch
from torchvision import transforms
from matplotlib import pyplot as plt

from dataset import SatelliteDataset
from utils import extract_samples
from satadv import SatAdv

import config as cfg
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

import pdb

"""Load the dataset"""
dataset_transform = transforms.Compose([transforms.ToTensor()])
train_set = SatelliteDataset(cfg.TRAIN_PATH, transform=dataset_transform, device=device)

"""Extract samples of background"""
samples = extract_samples(train_set, 1, 3)

"""Plot the samples"""
# k = 0
# for sample in samples:
#     img = sample[0].cpu().clone().permute(1,2,0).numpy()
#     plt.imshow(img)
#     plt.savefig(f"results/img_{k}.jpg")
#     plt.close('all')
#     k += 1


"""Initialize the adversarial attacker"""
adv_net = SatAdv(cfg)

"""Print model details"""
for name, param in adv_net.named_parameters():
    if param.requires_grad:
        print(name)

"""Generate a sample synthetic image"""
x = adv_net.render_synthetic_image(adv_net.meshes[0], samples[0][0].permute(1,2,0))