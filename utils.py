from torch import nn

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