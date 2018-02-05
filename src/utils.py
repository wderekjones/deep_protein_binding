import time
import torch
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, r2_score
from tqdm import tqdm

def collate_fn(batch):
    return batch


def train_step(model, batch, loss_fn, use_cuda=False):
    preds = []
    targets = []
    losses = []
    r2s = []

    #make sure model is in train mode
    model.train()
    start_train_clock = time.clock()

    for data in tqdm(batch, total=len(batch)):

        # Forward pass: compute output of the network by passing x through the model.
        y_pred = model(h=data["h"], g=data["g"])
        y_true = Variable(torch.from_numpy(data["target"]).float())
        if use_cuda: # TODO: add logic for cpu mode
            y_true = y_true.cuda()
        # Compute loss.
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        losses.append(loss.data.cpu().numpy())
        targets.append(y_true.data.cpu().numpy())
        preds.append(y_pred.data.cpu().numpy())
        r2s.append(r2_score(y_true=y_true.data.cpu().numpy(), y_pred=y_pred.data.cpu().numpy()))

    stop_train_clock = time.clock()

    return {"loss": np.mean(losses), "r2": np.mean(r2s), "time": (stop_train_clock - start_train_clock)}


def validation_step(model, dataloader, loss_fn, use_cuda=False):
    # make sure model is evaluation mode
    model.eval()

    preds = []
    targets = []
    losses = []
    r2s = []

    start_clock = time.clock()
    for batch in tqdm(dataloader, total=len(dataloader)):
        for data in batch:

            # Forward pass: compute output of the network by passing x through the model.
            y_pred = model(h=data["h"], g=data["g"])
            y_true = Variable(torch.from_numpy(data["target"]).float())
            if use_cuda: #TODO: add logic for cpu mode
                y_true = y_true.cuda()
            # Compute loss.
            losses.append(loss_fn(y_pred, y_true).data.cpu().numpy())
            targets.append(y_true.data.cpu().numpy())
            preds.append(y_pred.data.cpu().numpy())
            r2s.append(r2_score(y_true=y_true.data.cpu().numpy(), y_pred=y_pred.data.cpu().numpy()))

    stop_clock = time.clock()

    # put the model back into training mode
    model.train()

    # return a dictionary objects containing the validation metrics
    return {"loss": np.mean(losses), "r2": np.mean(r2s), "time": (stop_clock - start_clock)}


def update_scalars(writer, train_dict, val_dict, step):
    if train_dict is not None:
        writer.add_scalar("train_loss", float(train_dict["loss"]), step)
        writer.add_scalar("train_r2", float(train_dict["r2"]), step)
        writer.add_scalar("train_time", float(train_dict["time"]), step)
    if val_dict is not None:
        writer.add_scalar("val_loss", float(val_dict["loss"]), step)
        writer.add_scalar("val_r2", float(val_dict["r2"]), step)
        writer.add_scalar("val_time", float(val_dict["time"]), step)
