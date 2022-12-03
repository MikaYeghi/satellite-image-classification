import os
from torch import nn
from matplotlib import pyplot as plt

def make_train_step(model, loss_fn, optimizer):
    def train_step(images_batch, labels_batch):
        activation = nn.Sigmoid()
        model.train()
        yhat = model(images_batch)
        yhat = activation(yhat)
        loss = loss_fn(yhat, labels_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss.item()
    return train_step

def plot_training_info(train_losses, val_losses, save_dir='results'):
    plt.figure()
    plt.subplot(211)
    plt.plot(train_losses, 'b')
    plt.grid(True)
    plt.xlabel("Iteration number")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Training loss")
    plt.subplot(212)
    plt.plot(val_losses, 'b')
    plt.grid(True)
    plt.xlabel("Iteration number")
    plt.ylabel("Cross Entropy Loss")
    plt.title("Validation loss")
    
    save_dir = os.path.join(save_dir, "results.jpg")
    print(f"Saving the graphs to {save_dir}")
    plt.savefig(save_dir)