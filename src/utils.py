import time
import torch
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, r2_score
from tqdm import tqdm

def collate_fn(batch):
    return batch


def train_step(model, batch, loss_fn, target_list, use_cuda=False):
    preds = []
    targets = []
    losses = []

    #make sure model is in train mode
    model.train()
    start_train_clock = time.clock()

    for data in tqdm(batch, total=len(batch)):

        # Forward pass: compute output of the network by passing x through the model.
        y_pred = model(h=data["h"], g=data["g"])
        # Compute loss.
        loss = Variable(torch.zeros(1).float(), requires_grad=False)
        for pred, target in zip(y_pred, target_list):
            if use_cuda:
                y_true = Variable(torch.from_numpy(data[target]).float().cuda())
                loss = loss.cuda()
                loss += loss_fn(pred, y_true)
                losses.append(loss.data.cpu().numpy().ravel())
                targets.append(y_true.data.cpu().numpy().ravel())
                preds.append(pred.data.cpu().numpy().ravel())
            else:
                y_true =  Variable(torch.from_numpy(data[target])).float()
                loss += loss_fn(pred,y_true)
                losses.append(loss.data.numpy().ravel())
                targets.append(y_true.data.numpy().ravel())
                preds.append(pred.data.numpy().ravel())
        loss.backward()

    r2 = r2_score(y_true=np.ravel(targets), y_pred=np.ravel(preds))

    stop_train_clock = time.clock()

    return {"loss": np.mean(losses), "r2": r2, "time": (stop_train_clock - start_train_clock)}


def validation_step(model, batch, loss_fn,target_list, use_cuda=False):
    # make sure model is evaluation mode
    model.eval()

    preds = []
    targets = []
    losses = []

    start_clock = time.clock()

    for data in tqdm(batch, total=len(batch)):

        # Forward pass: compute output of the network by passing x through the model.
        y_pred = model(h=data["h"], g=data["g"])
        # Compute loss.
        loss = Variable(torch.zeros(1).float(), requires_grad=False)

        for pred, target in zip(y_pred, target_list):
            if use_cuda:
                y_true = Variable(torch.from_numpy(data[target]).float().cuda())
                loss = loss.cuda()
                loss += loss_fn(pred, y_true)
                losses.append(loss.data.cpu().numpy().ravel())
                targets.append(y_true.data.cpu().numpy().ravel())
                preds.append(pred.data.cpu().numpy().ravel())
            else:
                y_true =  Variable(torch.from_numpy(data[target])).float()
                loss += loss_fn(pred,y_true)
                losses.append(loss.data.numpy().ravel())
                targets.append(y_true.data.numpy().ravel())
                preds.append(pred.data.numpy().ravel())

    stop_clock = time.clock()
    r2 = r2_score(y_true=np.ravel(targets), y_pred=np.ravel(preds))

    # put the model back into training mode
    model.train()

    # return a dictionary objects containing the validation metrics
    return {"loss": np.mean(losses), "r2": r2, "time": (stop_clock - start_clock)}


def update_scalars(writer, train_dict, val_dict, step):
    if train_dict is not None:
        writer.add_scalar("train_loss", float(train_dict["loss"]), step)
        writer.add_scalar("train_r2", float(train_dict["r2"]), step)
        writer.add_scalar("train_time", float(train_dict["time"]), step)
    if val_dict is not None:
        writer.add_scalar("val_loss", float(val_dict["loss"]), step)
        writer.add_scalar("val_r2", float(val_dict["r2"]), step)
        writer.add_scalar("val_time", float(val_dict["time"]), step)
