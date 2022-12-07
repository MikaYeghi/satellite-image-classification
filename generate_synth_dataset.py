pass

from dataset import SatelliteDataset

"""Load the dataset"""
dataset_transform = transforms.Compose([transforms.ToTensor()])
train_set = SatelliteDataset(cfg.TRAIN_PATH, transform=train_transform, device=device)

