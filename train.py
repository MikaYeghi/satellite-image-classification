import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.nn import BCELoss
from torchvision import transforms

from dataset import SatelliteDataset
from utils import make_train_step

import config as cfg
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

import pdb

"""Load the data set"""
train_transform = transforms.Compose([transforms.ToTensor()])
train_path = cfg.TRAIN_PATH
train_set = SatelliteDataset(train_path, transform=train_transform, device=device)

"""Create the dataloader"""
train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE)

"""Initialize the model"""
model = models.resnet101(pretrained=True)
# model.fc.in_features = model.fc.in_features
# model.fc.out_features = cfg.NUM_CLASSES
model.fc = torch.nn.Linear(2048, 1, device=device, dtype=torch.float32)
if cfg.MODEL_WEIGHTS:
    pass
model.to(device)

"""Loss function and optimizer"""
loss_fn = BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)

"""Training"""
train_step = make_train_step(model, loss_fn, optimizer)
train_losses = []

for epoch in range(cfg.N_EPOCHS):
    t = tqdm(train_loader, desc=f"Epoch #{epoch + 1}")
    for images_batch, labels_batch in t:
        labels_batch = labels_batch.unsqueeze(1).float().to(device)

        loss = train_step(images_batch, labels_batch)
        
        train_losses.append(loss)
        
        t.set_description(f"Epoch: #{epoch + 1}. Loss: {round(loss, 8)}")