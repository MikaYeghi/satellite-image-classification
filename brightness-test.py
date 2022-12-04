import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.models as models
from torch import nn
from torch.nn import BCELoss
from torchvision import transforms
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from dataset import SatelliteDataset
from utils import make_train_step, plot_training_info, get_F1_stats

import config as cfg
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

import pdb

"""Transforms"""
test_transform = transforms.Compose([transforms.ToTensor()])

"""Initialize the model"""
model = models.resnet101(pretrained=True)
model.fc = torch.nn.Linear(2048, 1, device=device, dtype=torch.float32)
if cfg.MODEL_WEIGHTS:
    print(f"Loading model weights from {cfg.MODEL_WEIGHTS}")
    model.load_state_dict(torch.load(cfg.MODEL_WEIGHTS))
model.to(device)

"""Inference"""
print("Running brightness tests.")
accuracies = []
F1_scores = []
for brightness in cfg.BRIGHTNESS_LEVELS:
    print(f"Brightness level: {brightness}.")
    total_count = 0
    correct_count = 0
    TP = 0
    FP = 0
    FN = 0
    test_set = SatelliteDataset(cfg.TEST_PATH, transform=test_transform, device=device)
    test_set.set_brightness(brightness)
    test_set.leave_fraction_of_negatives(0.025)
    test_loader = DataLoader(test_set, batch_size=cfg.BATCH_SIZE)
    print(test_set.details())
    
    with torch.no_grad():
        activation = nn.Sigmoid()
        for images_batch, labels_batch in tqdm(test_loader):
            labels_batch = labels_batch.unsqueeze(1).float().to(device)

            model.eval()
            preds = activation(model(images_batch))

            # Convert to labels
            preds = (preds > 0.5).float()
            
            # Get the statistics for the F1-score
            TP_, FP_, FN_ = get_F1_stats(preds, labels_batch)

            # Compute the number of total and correct predictions
            correct_count_ = sum((labels_batch == preds).int()).item()
            total_count_ = len(preds)

            # Update the overall values
            total_count += total_count_
            correct_count += correct_count_
            TP += TP_
            FP += FP_
            FN += FN_

        accuracy = correct_count / total_count
        F1 = 2 * TP / (2 * TP + FP + FN)
        F1_scores.append(F1)
        accuracies.append(accuracy)
        print(f"Brightness: {brightness}. Accuracy: {round(100 * accuracy, 2)}%. F1-score: {round(100 * F1, 2)}%.")

# Plot the info
plt.figure()
plt.plot(cfg.BRIGHTNESS_LEVELS, F1_scores, 'b')
plt.grid(True)
plt.xlabel("Brightness level")
plt.ylabel("Accuracy")
plt.title("Model accuracy vs brightness level")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
save_dir = "results/brightness-test.jpg"
print(f"Saving brightness test results to {save_dir}")
plt.savefig(save_dir)