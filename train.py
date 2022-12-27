import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.models as models
from torch import nn
from sklearn.model_selection import train_test_split
import seaborn as sn
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from pathlib import Path
from torchvision.utils import save_image

from dataset import SatelliteDataset
from utils import make_train_step, plot_training_info, get_F1_stats, create_model
from evaluator import SatEvaluator
from transforms import SatTransforms
from losses import BCELoss, FocalLoss

import config as cfg
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

import pdb

"""Load the data set"""
transforms = SatTransforms()
train_transform = transforms.get_train_transforms()
test_transform = transforms.get_test_transforms()
train_set = SatelliteDataset(cfg.TRAIN_PATH, transform=train_transform, device=device)
test_set = SatelliteDataset(cfg.TEST_PATH, transform=test_transform, device=device)
print(f"Train set. {train_set.details()}\nTest set. {test_set.details()}")

"""Create the dataloader"""
train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE)
test_loader = DataLoader(test_set, batch_size=cfg.BATCH_SIZE)

"""Initialize the model"""
model = create_model(cfg, device)

"""Loss function, optimizer and evaluator"""
loss_fn = FocalLoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
evaluator = SatEvaluator(device=device, pos_label=0, results_dir=cfg.RESULTS_DIR)

"""Training"""
train_step = make_train_step(model, loss_fn, optimizer)

if not cfg.EVAL_ONLY:
    for epoch in range(cfg.N_EPOCHS):
        t = tqdm(train_loader, desc=f"Epoch #{epoch + 1}")
        for images_batch, labels_batch in t:
            labels_batch = labels_batch.unsqueeze(1).float().to(device)

            loss = train_step(images_batch, labels_batch)

            evaluator.record_train_loss(loss)

            t.set_description(f"Epoch: #{epoch + 1}. Loss: {round(loss, 8)}")

        # Save the intermediate model
        Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
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
                    evaluator.record_test_loss(val_loss.item())

                    t.set_description(f"Epoch: #{epoch + 1}. Validation loss: {round(val_loss.item(), 4)}.")

    model_save_dir = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    print(f"Saving the final model to {model_save_dir}")
    torch.save(model.state_dict(), model_save_dir)
    evaluator.plot_training_info()

print("Running inference.")
total_preds = torch.empty(size=(0, 1), device=device)
total_gt = torch.empty(size=(0, 1), device=device)
with torch.no_grad():
    activation = nn.Sigmoid()
    for images_batch, labels_batch in tqdm(test_loader):
        labels_batch = labels_batch.unsqueeze(1).float().to(device)

        model.eval()
        preds = activation(model(images_batch))
        
        # Convert to labels
        preds = (preds > 0.5).float()
        
        # Save FP-FN if needed
        if cfg.FP_FN_analysis:
            evaluator.save_FP_FN(preds, labels_batch, images_batch)

        # Record the predictions
        evaluator.record_preds_gt(preds, labels_batch)
    
    accuracy = evaluator.evaluate_accuracy()
    F1 = evaluator.evaluate_f1()
    
    # Plot the confusion matrix
    evaluator.plot_confmat()
    
    # Print the results
    print(f"Accuracy: {round(100 * accuracy, 2)}%. F1-score: {round(100 * F1, 2)}%.")