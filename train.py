import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.models as models
from torch import nn
from torch.nn import BCELoss
from torchvision import transforms
from sklearn.model_selection import train_test_split
import seaborn as sn
import pandas as pd
from matplotlib import pyplot as plt

from dataset import SatelliteDataset
from utils import make_train_step, plot_training_info, get_F1_stats

import config as cfg
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

import pdb

"""Load the data set"""
train_transform = transforms.Compose([transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])
train_set = SatelliteDataset(cfg.TRAIN_PATH, transform=train_transform, device=device)
test_set = SatelliteDataset(cfg.TEST_PATH, transform=test_transform, device=device)
# test_set.leave_fraction_of_negatives(0.025)
# train_set.augment_brightness(cfg.BRIGHTNESS_LEVELS)
print(f"Train set. {train_set.details()}\nTest set. {test_set.details()}")

"""Create the dataloader"""
train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE)
test_loader = DataLoader(test_set, batch_size=cfg.BATCH_SIZE)

"""Initialize the model"""
model = models.resnet101(pretrained=True)
model.fc = torch.nn.Linear(2048, 1, device=device, dtype=torch.float32)
if cfg.MODEL_WEIGHTS:
    print(f"Loading model weights from {cfg.MODEL_WEIGHTS}")
    model.load_state_dict(torch.load(cfg.MODEL_WEIGHTS))
model.to(device)

"""Loss function and optimizer"""
loss_fn = BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)

"""Training"""
train_step = make_train_step(model, loss_fn, optimizer)
train_losses = []
test_losses = []

if not cfg.EVAL_ONLY:
    for epoch in range(cfg.N_EPOCHS):
        t = tqdm(train_loader, desc=f"Epoch #{epoch + 1}")
        for images_batch, labels_batch in t:
            labels_batch = labels_batch.unsqueeze(1).float().to(device)

            loss = train_step(images_batch, labels_batch)

            train_losses.append(loss)

            t.set_description(f"Epoch: #{epoch + 1}. Loss: {round(loss, 8)}")

        # Save the intermediate model
        torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, f"model_{epoch + 1}.pth"))

        if (epoch + 1) % cfg.VAL_FREQ == 0:
            print("Running validation...")
            with torch.no_grad():
                activation = nn.Sigmoid()
                t = tqdm(test_loader)
                for images_batch, labels_batch in t:
                    labels_batch = labels_batch.unsqueeze(1).float().to(device)

                    model.eval()
                    preds = activation(model(images_batch))

                    val_loss = loss_fn(preds, labels_batch)
                    test_losses.append(val_loss.item())

                    t.set_description(f"Epoch: #{epoch + 1}. Validation loss: {round(val_loss.item(), 4)}.")

    model_save_dir = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    print(f"Saving the final model to {model_save_dir}")
    torch.save(model.state_dict(), model_save_dir)
    plot_training_info(train_losses, test_losses)

print("Running inference.")
total_count = 0
correct_count = 0
TP = 0
FP = 0
FN = 0
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
    TN = total_count - TP - FN - FP
    confusion_matrix = torch.tensor([
            [TP / (TP + FN), FN / (TP + FN)],
            [FP / (TN + FP), TN / (TN + FP)]
        ])
    confusion_matrix = pd.DataFrame(confusion_matrix, 
                                    index=["Actual positive", "Actual negative"],
                                    columns=["Predicted positive", "Predicted negative"]
    )
    # Plot the confusion matrix
    plt.figure()
    sn.heatmap(confusion_matrix, annot=True, cmap="Blues")
    plt.savefig("results/confmat.jpg")
    
    # Print the results
    print(f"Accuracy: {round(100 * accuracy, 2)}%. F1-score: {round(100 * F1, 2)}%.")