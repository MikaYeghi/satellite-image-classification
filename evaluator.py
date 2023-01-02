import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from matplotlib import pyplot as plt
from torchvision.utils import save_image
from pathlib import Path
import pandas as pd
import seaborn as sn
import os

class SatEvaluator():
    def __init__(self, device='cuda:0', pos_label=0, save_dir="results"):
        self.device = device
        self.pos_label = pos_label
        self.save_dir = save_dir
        
        self.total_preds = torch.empty(size=(0, 1), device=device)
        self.total_gt = torch.empty(size=(0, 1), device=device)
        
        self.train_losses = []
        self.test_losses = []
        
        # FP-FN analysis
        self.FP_counter = 0
        self.FN_counter = 0
        self.FP_save_dir = os.path.join(save_dir, "fp-fn-analysis", "FP")
        self.FN_save_dir = os.path.join(save_dir, "fp-fn-analysis", "FN")
        Path(self.FP_save_dir).mkdir(parents=True, exist_ok=True) # create the directory if necessary
        Path(self.FN_save_dir).mkdir(parents=True, exist_ok=True) # create the directory if necessary
    
    def record_preds_gt(self, preds, gt):
        self.total_preds = torch.cat((preds, self.total_preds))
        self.total_gt = torch.cat((gt, self.total_gt))
        
    def record_train_loss(self, train_loss):
        if torch.is_tensor(train_loss):
            train_loss = train_loss.item()
        self.train_losses.append(train_loss)
        
    def record_test_loss(self, test_loss):
        if torch.is_tensor(test_loss):
            test_loss = test_loss.item()
        self.test_losses.append(test_loss)
        
    def evaluate_accuracy(self):
        return accuracy_score(self.total_gt.cpu(), self.total_preds.cpu())
    
    def evaluate_f1(self):
        return f1_score(self.total_gt.cpu(), self.total_preds.cpu(), pos_label=self.pos_label)
    
    def evaluate_confmat(self):
        return confusion_matrix(self.total_gt.cpu(), self.total_preds.cpu(), normalize='true')
    
    def plot_training_info(self):
        plt.figure(figsize=(6.4, 7.5))
        plt.subplot(211)
        plt.plot(self.train_losses, 'b')
        plt.xlabel("Iteration number")
        plt.ylabel("Cross Entropy Loss")
        plt.title("Training loss")
        if len(self.train_losses) > 0:
            plt.yscale('log')
        plt.grid(True)
        plt.subplot(212)
        plt.plot(self.test_losses, 'b')
        plt.xlabel("Iteration number")
        plt.ylabel("Cross Entropy Loss")
        plt.title("Validation loss")
        if len(self.test_losses) > 0:
            plt.yscale('log')
        plt.grid(True)
        
        plt.tight_layout()

        save_dir = os.path.join(self.save_dir, "trainval_curves.jpg")
        print(f"Saving the graphs to {save_dir}")
        plt.savefig(save_dir)
        plt.close('all')
    
    def plot_confmat(self):
        confusion_matrix = pd.DataFrame(self.evaluate_confmat(), 
                                    index=["Actual positive", "Actual negative"],
                                    columns=["Predicted positive", "Predicted negative"]
        )
        plt.figure()
        sn.heatmap(confusion_matrix, annot=True, cmap="Blues")
        plt.savefig(os.path.join(self.save_dir, "confmat.jpg"))
        plt.close('all')
    
    def reset(self):
        self.total_preds = torch.empty(size=(0, 1), device=self.device)
        self.total_gt = torch.empty(size=(0, 1), device=self.device)
        
        self.train_losses = []
        self.test_losses = []
        
    def save_FP_FN(self, preds, labels_batch, images_batch):
        # Save FP and FN images
        for i in range(len(preds)):
            if preds[i] == 0 and labels_batch[i] == 1:
                # FP
                img = images_batch[i]
                save_image(img, os.path.join(self.FP_save_dir, f"image_{self.FP_counter}.jpg"))
                self.FP_counter += 1
            elif preds[i] == 1 and labels_batch[i] == 0:
                # FN
                img = images_batch[i]
                save_image(img, os.path.join(self.FN_save_dir, f"image_{self.FN_counter}.jpg"))
                self.FN_counter += 1