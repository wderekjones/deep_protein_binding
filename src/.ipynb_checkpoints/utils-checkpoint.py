import time
import torch
import numpy as np
import argparse
from torch.autograd import Variable
from sklearn.metrics import r2_score
from tqdm import tqdm
from loss import MultiTaskHomoscedasticLoss, MultiTaskNormalLoss, MultiTaskWeightedLoss


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-D', type=str, help="path to dataset", default="/u/vul-d1/scratch/wdjo224/data/deep_protein_binding/kinase_no_duplicates_with_smiles.csv")
    parser.add_argument('-c', type=str, help="path to corrupted inputs", default="/u/vul-d1/scratch/wdjo224/data/deep_protein_binding/corrupt_inputs.csv")
    parser.add_argument("--T", type=int, help="number of message passing steps", default=3)
    parser.add_argument("--n_workers", type=int, help="number of workers to use in data loader", default=0)
    parser.add_argument("--batch_size", type=int, help="batch size to use in data loader", default=1)
    parser.add_argument("--n_epochs", type=int, help="number of training epochs", default=1)
    parser.add_argument("--use_cuda", type=bool, help="indicates whether to use gpu", default=False)
    parser.add_argument("--lr", type=float, help="learning rate to use for training", default=1e-3)
    parser.add_argument("--p", type=float, help="value for p in dropout layer", default=0.5)
    parser.add_argument("--exp_name", type=str, help="name of the experiment", default="debug")
    parser.add_argument("--scale", type=str, help="type of scaling to use (norm or std)", default="std")
    parser.add_argument("--target_list", type=str, nargs="+", help="list of targets to train the network with", default=["Uc","Ui","Vx","MLOGP","SAtot"])
    parser.add_argument("--target_file", type=str, help="path to file containing list of targets", default=None)
    parser.add_argument("--model_path", type=str, help="path to model file to continue training", default=None)
    parser.add_argument("--train_idxs", type=str, help="path to train indexes", default="/u/vul-d1/scratch/wdjo224/deep_protein_binding/src/train.npy")
    parser.add_argument("--test_idxs", type=str, help="path to test indexes", default="/u/vul-d1/scratch/wdjo224/deep_protein_binding/src/test.npy")
    parser.add_argument("--val_idxs", type=str, help="path to validation indexes", default="/u/vul-d1/scratch/wdjo224/deep_protein_binding/src/val.npy")
    parser.add_argument("--w_prior", type=str, help="prior distribution to use for multitask loss, unweighted by default", default=None)
    parser.add_argument("--loss", type=str, help="loss function to use (homoscedastic, normal, weighted) ", default="weighted")
    args = parser.parse_args()
    
    return args


def get_loss(args):
    loss_fn = None
    if args.loss == "homoscedastic":
        loss_fn = MultiTaskHomoscedasticLoss(n_tasks=len(args.target_list), prior=args.w_prior)
    elif args.loss == "normal":
        loss_fn = MultiTaskNormalLoss(n_tasks=len(args.target_list), prior=args.w_prior)
    elif args.loss == "weighted":
        loss_fn = MultiTaskWeightedLoss(n_tasks=len(args.target_list), prior=args.w_prior)
    else:
        raise Exception("loss function not implemented.")

    return loss_fn

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

    batch_dict = {key: {"batch_size": len(batch), "pred": [], "true": [], "loss": [], "r2": []} for key in target_list}
    start_clock = time.clock()

    for data in tqdm(batch, total=len(batch)):

        # Forward pass: compute output of the network by passing x through the model. Get N outputs as a result
        y_pred = model(h=data["h"], g=data["g"])

        for idx, (pred, target) in enumerate(zip(y_pred, target_list)):

            if use_cuda:
                y_true = Variable(torch.from_numpy(data[target]).float().cuda())
                batch_dict[target]["true"].append(y_true)
                batch_dict[target]["pred"].append(pred)

            else:
                y_true = Variable(torch.from_numpy(data[target])).float()
                batch_dict[target]["true"].append(y_true)
                batch_dict[target]["pred"].append(pred)

    for target in target_list:
        batch_dict[target]["r2"] = r2_score(y_true=torch.stack(batch_dict[target]["true"]).data.numpy(),
                                            y_pred=torch.stack(batch_dict[target]["pred"]).data.numpy())

    stop_clock = time.clock()

    loss = loss_fn(batch_dict)

    # return a dictionary objects containing the metrics
    return {"loss": loss, "target_dict": batch_dict, "time": (stop_clock - start_clock)}


def update_tensorboard(writer, train_dict, val_dict, step):
    if train_dict is not None:
        writer.add_scalar("train_loss", float(train_dict["loss"]), step)
        writer.add_scalar("train_time", float(train_dict["time"]), step)
        for target in train_dict["target_dict"].keys():
            writer.add_scalar("train_"+target+"_r2", float(np.mean(train_dict["target_dict"][target]["r2"])), step)
            writer.add_histogram("train_" + target + "_r2_hist", float(np.mean(train_dict["target_dict"][target]["r2"])), step)
            writer.add_scalar("train_" + target + "_loss", float(np.mean(train_dict["target_dict"][target]["loss"])), step)
            writer.add_histogram("train_" + target + "_loss_hist", float(np.mean(train_dict["target_dict"][target]["loss"])),step)
            writer.add_scalar("train_"+ target + "_loss_weight", train_dict["target_dict"][target]["loss_weight"], step)
            writer.add_histogram("train_"+ target + "_loss_weight_hist", train_dict["target_dict"][target]["loss_weight"], step)

    if val_dict is not None:
        writer.add_scalar("val_loss", float(val_dict["loss"]), step)
        writer.add_scalar("val_time", float(val_dict["time"]), step)
        for target in val_dict["target_dict"].keys():
            writer.add_scalar("val_"+ target+ "_r2", float(np.mean(val_dict["target_dict"][target]["r2"])), step)
            writer.add_histogram("val_"+ target+ "_r2_hist", float(np.mean(val_dict["target_dict"][target]["r2"])), step)
            writer.add_scalar("val_" + target + "_loss", float(np.mean(val_dict["target_dict"][target]["loss"])), step)
            writer.add_histogram("val_" + target + "_loss_hist", float(np.mean(val_dict["target_dict"][target]["loss"])), step)
            writer.add_scalar("val_"+ target + "_loss_weight", val_dict["target_dict"][target]["loss_weight"], step)
            writer.add_histogram("val_" + target + "_loss_weight_hist", val_dict["target_dict"][target]["loss_weight"], step)
