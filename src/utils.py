import time
import torch
import numpy as np
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, r2_score
from tqdm import tqdm

def collate_fn(batch):
    return batch


def train_step(model, batch, loss_fn, target_list, use_cuda=False):

    #make sure model is in train mode
    model.train()

    batch_dict = step(model=model, batch=batch, loss_fn=loss_fn, target_list=target_list, use_cuda=use_cuda)

    # update model gradients()
    batch_dict["loss"].backward()

    return batch_dict


def validation_step(model, batch, loss_fn,target_list, use_cuda=False):
    # make sure model is evaluation mode
    model.eval()

    batch_dict = step(model=model, batch=batch, loss_fn=loss_fn, target_list=target_list, use_cuda=use_cuda)

    # put the model back into training mode
    model.train()

    return batch_dict


def step(model, batch, loss_fn, target_list, use_cuda=False):

    batch_dict = {key: {"pred": [], "true": [], "loss": [], "r2": []} for key in target_list}

    start_clock = time.clock()

    for data in tqdm(batch, total=len(batch)):

        # Forward pass: compute output of the network by passing x through the model. Get N outputs as a result
        y_pred = model(h=data["h"], g=data["g"])

        for idx, (pred, target) in enumerate(zip(y_pred, target_list)):

            if use_cuda:
                y_true = Variable(torch.from_numpy(data[target]).float().cuda())
                # losses.append(loss.data.cpu().numpy().ravel())
                # batch_dict[target]["loss"].append(loss.data.cpu().numpy().ravel())
                batch_dict[target]["true"].append(y_true)
                batch_dict[target]["pred"].append(pred)

            else:
                y_true = Variable(torch.from_numpy(data[target])).float()
                # losses.append(loss.data.numpy().ravel())
                # batch_dict[target]["loss"].append(loss.data.numpy().ravel())
                batch_dict[target]["true"].append(y_true)
                batch_dict[target]["pred"].append(pred)

    for target in target_list:
        batch_dict[target]["r2"] = r2_score(y_true=torch.stack(batch_dict[target]["true"]).data.numpy(),
                                            y_pred=torch.stack(batch_dict[target]["pred"]).data.numpy())

    stop_clock = time.clock()

    loss = loss_fn(batch_dict)

    # return a dictionary objects containing the metrics
    return ({"loss": loss, "target_dict": batch_dict, "time": (stop_clock - start_clock)})


def update_scalars(writer, train_dict, val_dict, step):
    if train_dict is not None:
        writer.add_scalar("train_loss", float(train_dict["loss"]), step)
        writer.add_scalar("train_time", float(train_dict["time"]), step)
        for target in train_dict["target_dict"].keys():
            writer.add_scalar("train_"+target+"_r2", float(np.mean(train_dict["target_dict"][target]["r2"])), step)
    if val_dict is not None:
        writer.add_scalar("val_loss", float(val_dict["loss"]), step)
        writer.add_scalar("val_time", float(val_dict["time"]), step)
        for target in val_dict["target_dict"].keys():
            writer.add_scalar("val_"+target+"_r2", float(np.mean(val_dict["target_dict"][target]["r2"])), step)
