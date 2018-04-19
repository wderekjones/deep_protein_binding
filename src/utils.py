import torch
import numpy as np
import argparse
from src.loss import MultiTaskHomoscedasticLoss, MultiTaskNormalLoss, MultiTaskWeightedLoss, MultiTaskBCELoss


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="random number seed", default=0)
    parser.add_argument('-D', type=str, help="path to dataset", default="/scratch/wdjo224/data/deep_protein_binding/kinase_no_duplicates_with_smiles.csv")
    parser.add_argument('-c', type=str, help="path to corrupted inputs", default="/scratch/wdjo224/data/deep_protein_binding/corrupt_inputs.csv")
    parser.add_argument("--T", type=int, help="number of message passing steps", default=3)
    parser.add_argument("--n_workers", type=int, help="number of workers to use in data loader", default=0)
    parser.add_argument("--n_train_process", type=int, help="number of processes to use for training", default=1)
    parser.add_argument("--n_test_process", type=int, help="number of processes to use for testing", default=1)
    parser.add_argument("--pid", type=str, help="training pid")
    parser.add_argument("--batch_size", type=int, help="batch size to use in data loader", default=1)
    parser.add_argument("--n_epochs", type=int, help="number of training epochs", default=1)
    parser.add_argument("--use_cuda", type=bool, help="indicates whether to use gpu", default=False)
    parser.add_argument("--lr", type=float, help="learning rate to use for training", default=1e-3)
    parser.add_argument("--momentum", type=float, help="momentum term", default=0.9)
    parser.add_argument("--p", type=float, help="value for p in dropout layer", default=0.5)
    parser.add_argument("--exp_name", type=str, help="name of the experiment", default="debug")
    parser.add_argument("--scale", type=str, help="type of scaling to use (norm or std)", default=None)
    parser.add_argument("--target", type=str, help="target to train the network with", default=None)
    parser.add_argument("--model_path", type=str, help="path to model file", default=None)
    parser.add_argument("--train_idxs", type=str, help="path to train indexes", default="/scratch/wdjo224/deep_protein_binding/src/train.npy")
    parser.add_argument("--test_idxs", type=str, help="path to test indexes", default="/scratch/wdjo224/deep_protein_binding/src/test.npy")
    parser.add_argument("--val_idxs", type=str, help="path to validation indexes", default="/scratch/wdjo224/deep_protein_binding/src/val.npy")
    parser.add_argument("--w_prior", type=str, help="prior distribution to use for multitask loss, unweighted by default", default=None)
    parser.add_argument("--loss", type=str, help="loss function to use (homoscedastic, normal, weighted) ", default="weighted")
    parser.add_argument("--output_type", type=str, help="specify regression (regress) or classification output activation", default="regress")
    parser.add_argument("--output_dim", type=int, help="output dimension", default=1)
    parser.add_argument("--readout_dim", type=int, help="size readout output", default=128)
    parser.add_argument("--opt", type=str, help="optimizer to use during training", default="adam")
    parser.add_argument("--epoch", type=int, help="epoch from which to load weights", default=4)

    return parser


def get_loss(args):
    if args.loss == "homo":
        return MultiTaskHomoscedasticLoss(n_tasks=1, prior=args.w_prior)
    elif args.loss == "normal":
        return MultiTaskNormalLoss(n_tasks=1)
    elif args.loss == "weighted":
        return MultiTaskWeightedLoss(n_tasks=1, prior=args.w_prior)
    elif args.loss == "bce":
        return MultiTaskBCELoss(n_tasks=1, prior=args.w_prior)
    else:
        return None


def get_opt(args):
    if args.opt == "adam":
        return torch.optim.Adam
    elif args.opt == "sgd":
        return torch.optim.SGD
    else:
        return None


def collate_fn(batch):
    return batch


def update_tensorboard(writer, train_dict, val_dict, step):
    if train_dict is not None:
        writer.add_scalar("train_loss", float(train_dict["loss"]), step)
        writer.add_scalar("train_time", float(train_dict["time"]), step)
        for target in train_dict["batch_dict"].keys():
            writer.add_scalar("train_" + target + "_loss", float(np.mean(train_dict["batch_dict"][target]["loss"])),
                              step)
            for metric in train_dict["batch_dict"][target]["metrics"].keys():
                writer.add_scalar("train_"+target+"_"+str(metric), float(np.mean(train_dict["batch_dict"][target]["metrics"][metric])), step)

# use histograms for validation instead?

    if val_dict is not None:
        writer.add_scalar("val_loss", float(val_dict["loss"]), step)
        writer.add_scalar("val_time", float(val_dict["time"]), step)
        for target in val_dict["batch_dict"].keys():
            writer.add_scalar("val_" + target + "_loss", float(np.mean(val_dict["batch_dict"][target]["loss"])), step)
            writer.add_scalar("val_" + target + "_loss_weight", val_dict["batch_dict"][target]["loss_weight"], step)
            for metric in val_dict["batch_dict"][target]["metrics"].keys():
                writer.add_scalar("val_"+ target+ "_"+str(metric), float(np.mean(val_dict["batch_dict"][target]["metrics"][metric])), step)
