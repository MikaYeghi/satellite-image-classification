import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision.models as models
from torch import nn
from torch.nn import BCELoss
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sn

from dataset import SatelliteDataset
from utils import make_train_step, plot_training_info, get_F1_stats, create_model
from evaluator import SatEvaluator
from transforms import SatTransforms

import config as cfg
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

import pdb

"""Transforms"""
transforms = SatTransforms()
test_transform = transforms.get_test_transforms()

"""Initialize the model"""
model = create_model(cfg, device)
evaluator = SatEvaluator(device=device, pos_label=0, results_dir=cfg.RESULTS_DIR)

"""Inference"""
print("Running brightness tests.")
accuracies = []
F1_scores = []
confusion_matrices = []
for brightness in cfg.BRIGHTNESS_LEVELS:
    print(f"Brightness level: {brightness}.")
    test_set = SatelliteDataset(cfg.TEST_PATH, transform=test_transform, device=device)
    test_set.set_brightness(brightness)
    # test_set.leave_fraction_of_negatives(0.025)
    test_loader = DataLoader(test_set, batch_size=cfg.BATCH_SIZE)
    print(test_set.details())
    evaluator.reset()
    
    with torch.no_grad():
        activation = nn.Sigmoid()
        for images_batch, labels_batch in tqdm(test_loader):
            labels_batch = labels_batch.unsqueeze(1).float().to(device)

            model.eval()
            preds = activation(model(images_batch))

            # Convert to labels
            preds = (preds > 0.5).float()
            
            # Record the predictions
            evaluator.record_preds_gt(preds, labels_batch)

        accuracy = evaluator.evaluate_accuracy()
        F1 = evaluator.evaluate_f1()
        F1_scores.append(F1)
        accuracies.append(accuracy)
        
        # Generate the confusion matrix
        confusion_matrix = evaluator.evaluate_confmat()
        confusion_matrix = pd.DataFrame(confusion_matrix, 
                                        index=["Predicted positive", "Predicted negative"],
                                        columns=["Actual positive", "Actual negative"]
        )
        confusion_matrices.append(confusion_matrix)
        print(f"Brightness: {brightness}. Accuracy: {round(100 * accuracy, 2)}%. F1-score: {round(100 * F1, 2)}%.")

# Plot the info
plt.figure()
plt.plot(cfg.BRIGHTNESS_LEVELS, F1_scores, 'b')
plt.grid(True)
plt.xlabel("Brightness level")
plt.ylabel("F1 score")
plt.title("F1 score vs brightness level")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
save_dir = "results/brightness-test.jpg"
print(f"Saving brightness test results to {save_dir}")
plt.savefig(save_dir)
plt.close('all')

# Plot confusion matrices
save_dir = "results/confmats.jpg"
print(f"Saving confusion matrices to {save_dir}")
plt.figure(figsize=(15.0, 8.0))
for idx in range(len(confusion_matrices)):
    plt.subplot(2, 3, idx + 1)
    plt.title(f"Brightness: {cfg.BRIGHTNESS_LEVELS[idx]}")
    sn.heatmap(confusion_matrices[idx], annot=True, cmap="Blues")
plt.savefig(save_dir)