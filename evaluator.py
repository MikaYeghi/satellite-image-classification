import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

class SatEvaluator():
    def __init__(self, device='cuda:0', pos_label=0):
        self.device = device
        self.pos_label = pos_label
        
        self.total_preds = torch.empty(size=(0, 1), device=device)
        self.total_gt = torch.empty(size=(0, 1), device=device)
        
        self.train_losses = []
        self.val_losses = []
    
    def record_preds_gt(self, preds, gt):
        self.total_preds = torch.cat((preds, self.total_preds))
        self.total_gt = torch.cat((gt, self.total_gt))
        
    def record_train_loss(self, train_loss):
        self.train_losses.append(train_loss)
        
    def record_test_loss(self, test_loss):
        self.test_losses.append(test_loss)
        
    def evaluate_accuracy(self):
        return accuracy_score(self.total_gt.cpu(), self.total_preds.cpu())
    
    def evaluate_f1(self):
        return f1_score(self.total_gt.cpu(), self.total_preds.cpu(), pos_label=self.pos_label)
    
    def evaluate_confmat(self):
        return confusion_matrix(self.total_gt.cpu(), self.total_preds.cpu(), normalize='true')