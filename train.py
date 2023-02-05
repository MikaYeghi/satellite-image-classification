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
from torch.utils.tensorboard import SummaryWriter

from dataset import SatelliteDataset
from utils import make_train_step, plot_training_info, get_F1_stats, create_model, save_checkpoint, load_checkpoint
from evaluator import SatEvaluator
from transforms import SatTransforms
from losses import BCELoss, FocalLoss

import config as cfg
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

from logger import get_logger
logger = get_logger("Train logger")

import pdb
from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn', force=True)
except RuntimeError:
    logger.warning("NOTE: UNKNOWN THING DIDN'T WORK.")

def do_train(train_loader, test_loader, evaluator, model, loss_fn, train_step, scheduler, writer, start_epoch, iter_counter):
    if start_epoch >= cfg.N_EPOCHS:
        logger.info("Early stopping training.")
        return
    for epoch in range(start_epoch, cfg.N_EPOCHS):
        t = tqdm(train_loader, desc=f"Epoch #{epoch + 1}")
        for images_batch, labels_batch in t:
            labels_batch = labels_batch.unsqueeze(1).float().to(device)

            loss = train_step(images_batch, labels_batch)
            
            writer.add_scalar("Training loss", loss, iter_counter)
            writer.add_scalar("LR", scheduler.get_last_lr()[-1], iter_counter)
            evaluator.record_train_loss(loss)

            t.set_description(f"Epoch: #{epoch + 1}. Loss: {round(loss, 8)}")
            
            iter_counter += 1

        # Save the intermediate model
        save_checkpoint(cfg, model, epoch, iter_counter, is_final=False)

        if (epoch + 1) % cfg.VAL_FREQ == 0:
            logger.info("Running validation...")
            with torch.no_grad():
                activation = nn.Sigmoid()
                t = tqdm(test_loader)
                for images_batch, labels_batch in t:
                    labels_batch = labels_batch.unsqueeze(1).float().to(device)

                    model.eval()
                    preds = activation(model(images_batch))

                    val_loss = loss_fn(preds, labels_batch)
                    writer.add_scalar("Validation loss", val_loss.item(), iter_counter)
                    evaluator.record_test_loss(val_loss.item())

                    t.set_description(f"Epoch: #{epoch + 1}. Validation loss: {round(val_loss.item(), 4)}.")
        # scheduler.step()
    save_checkpoint(cfg, model, epoch, iter_counter, is_final=True)
    evaluator.plot_training_info()

def do_test(test_loader, model, evaluator):
    logger.info("Running inference.")
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
        results_text = f"Accuracy: {round(100 * accuracy, 2)}%.\nF1-score: {round(100 * F1, 2)}%."
        logger.info(results_text)
        with open(os.path.join(cfg.OUTPUT_DIR, "results.txt"), 'w') as f:
            f.write(results_text)
    
if __name__ == '__main__':
    """Load the data set"""
    transforms = SatTransforms()
    train_transform = transforms.get_train_transforms()
    test_transform = transforms.get_test_transforms()
    train_set = SatelliteDataset(cfg.TRAIN_PATH, transform=train_transform, device=device)
    test_set = SatelliteDataset(cfg.TEST_PATH, transform=test_transform, device=device)
    logger.info(f"Train set. {train_set.details()}")
    logger.info(f"Test set. {test_set.details()}")
    
    """Create the dataloader"""
    train_loader = DataLoader(train_set, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_DATALOADER_WORKERS, shuffle=cfg.SHUFFLE)
    test_loader = DataLoader(test_set, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_DATALOADER_WORKERS, shuffle=cfg.SHUFFLE)

    """Initialize the model"""
    # model = create_model(cfg, device)
    start_epoch, iter_counter, model = load_checkpoint(cfg, device)

    """Loss function, optimizer, scheduler and evaluator"""
    loss_fn = FocalLoss(alpha=cfg.FOCAL_LOSS['alpha'])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.LR_GAMMA)
    evaluator = SatEvaluator(device=device, pos_label=0, save_dir=cfg.OUTPUT_DIR)

    """Training"""
    train_step = make_train_step(model, loss_fn, optimizer)
    writer = SummaryWriter(log_dir=cfg.LOG_DIR)
    
    # Make sure the save directory exists
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # Run training
    if not cfg.EVAL_ONLY:
        do_train(train_loader, test_loader, evaluator, model, loss_fn, train_step, scheduler, writer, start_epoch, iter_counter)

    # Run evaluation
    do_test(test_loader, model, evaluator)
    
    # Close the tensorboard logger
    writer.flush()
    writer.close()