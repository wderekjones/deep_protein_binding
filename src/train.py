import sys
sys.path.append("/scratch/wdjo224/deep_protein_binding")
import os
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from src.utils import get_parser, update_tensorboard, collate_fn, get_loss
from src.MoleculeDataset import MoleculeDatasetCSV
from tensorboardX import SummaryWriter

args = get_parser().parse_args()

root = "/scratch/wdjo224/deep_protein_binding"
experiment_path = root + "/" + "experiments" + "/" + args.exp_name
checkpoint_path = experiment_path + "/" + "checkpoints"
scalar_path = experiment_path + "/" + "scalars"
log_path = root + "/" + "logs" + "/" + args.exp_name

# create a driectory to store experiment data
if not os.path.exists(experiment_path):
    print("creating experiment path...")
    os.makedirs(experiment_path)

# dump the arguments to a file..had to do some weird things here to get things to work with eval script (temp)
pd.DataFrame(vars(args), index=[0]).to_csv(experiment_path+"/args.csv")
df = pd.read_csv(experiment_path+"/args.csv", index_col=0)
df = df.transpose().reset_index()["index"].apply(lambda x: "--"+x+"=").map(str) + \
df.transpose().reset_index()[0].map(str)
df = df.drop(df[df.str.contains("nan")].index)
pd.DataFrame(df).to_csv(experiment_path+"/args.csv")

# create a directory to store checkpoints
if not os.path.exists(checkpoint_path):
    print("creating checkpoint path...")
    os.makedirs(checkpoint_path)

# create a directory to store scalar data
if not os.path.exists(scalar_path):
    print("creating scalar path...")
    os.makedirs(scalar_path)

# create a path to write tensorboard logs
if not os.path.exists(log_path):
    print("creating tensorboard log path...")
    os.makedirs(log_path)


def train(rank, args, model):

    torch.manual_seed(args.seed + rank)

    writer = SummaryWriter(log_path + "/" + "p{}".format(os.getpid()))

    writer.add_text('args', str(sys.argv))
    writer.add_text("target", str(args.target))
    writer.add_text("pid", str(os.getpid()))

    print("loading data...")

    molecules = MoleculeDatasetCSV(csv_file=args.D, corrupt_path=args.c, target=args.target,
                                   scaling=args.scale)

    batch_size = args.batch_size
    train_idxs = np.fromfile(args.train_idxs, dtype=np.int)
    val_idxs = np.fromfile(args.val_idxs, dtype=np.int)

    molecule_loader_train = DataLoader(molecules, batch_size=batch_size, num_workers=0,
                                       collate_fn=collate_fn,
                                       sampler=SubsetRandomSampler(train_idxs))
    molecule_loader_val = DataLoader(molecules, batch_size=val_idxs.shape[0], num_workers=0,
                                     collate_fn=collate_fn,
                                     sampler=SubsetRandomSampler(val_idxs))

    loss_fn = get_loss(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    global_step = 0
    for epoch in range(0, args.n_epochs):
        global_step = train_epoch(rank=rank, epoch=epoch, global_step=global_step, model=model,
                    molecule_loader_train=molecule_loader_train, optimizer=optimizer, loss_fn=loss_fn, writer=writer)

        test_epoch(rank=rank, epoch=epoch, model=model, molecule_loader_val=molecule_loader_val, loss_fn=loss_fn,
                   writer=writer)

        print("Saving model checkpoint...")
        torch.save(model.state_dict(), checkpoint_path + "/" + "p{}".format(os.getpid()) + "_epoch{}".format(epoch)
                   + "_params.pth")

        # Output training metrics
        print("Saving training metrics")

        writer.export_scalars_to_json(scalar_path + "/" + "p{}".format(os.getpid()) + "_epoch{}".format(epoch) +
                                      "_scalars.json")

    writer.close()


def train_epoch(rank, epoch, global_step, model, molecule_loader_train, optimizer, loss_fn, writer):
    torch.manual_seed(args.seed+rank)
    pid = os.getpid()
    model.train()

    for batch_idx, batch in enumerate(molecule_loader_train):

        # zero the gradients for the next batch
        optimizer.zero_grad()

        # take a training step, i.e. process the mini-batch and accumulate gradients
        train_dict = model.train_step(batch=batch, loss_fn=loss_fn)

        # log the information to tensorboard
        update_tensorboard(writer=writer, train_dict=train_dict, val_dict=None, step=global_step)

        print("pid: {} \t epoch: {} \t step: {} \t train loss: {}".format(pid, epoch, global_step,
                                                                          train_dict["loss"].data))

        # update the model parameters
        optimizer.step()

        # increment the global step variable
        global_step += 1

    return global_step

def test_epoch(rank, epoch, model, molecule_loader_val, loss_fn, writer):
    torch.manual_seed(args.seed+rank)
    pid = os.getpid()

    for batch_idx, batch in enumerate(molecule_loader_val):

        # take a training step, i.e. process the mini-batch and accumulate gradients
        val_dict = model.test_step(batch=batch, loss_fn=loss_fn)

        # log the information to tensorboard
        update_tensorboard(writer=writer, train_dict=None, val_dict=val_dict, step=epoch)

        print("pid: {} \t epoch: {} \t val loss: {}".format(pid, epoch, val_dict["loss"].data))


def main():

    import torch.multiprocessing as mp
    mp.set_sharing_strategy("file_system")
    mp = mp.get_context("forkserver")
    from src.model import MPNN

    torch.manual_seed(args.seed)

    print("{:=^100}".format(' Train '))
    print("experiment: {}".format(args.exp_name))
    print("run parameters: {} \n".format(sys.argv))

    model_path = args.model_path

    print("instantiating model...")
    model = MPNN(T=args.T, p=args.p, target=args.target, output_type=args.output_type, output_dim=args.output_dim,
                 readout_dim=args.readout_dim)
    if model_path is not None:
        model.load_state_dict(model_path)

    model.share_memory()

    print(model)

    # Train the model
    print("Training Model...")

    processes = []
    for rank in range(args.n_train_process):
        p = mp.Process(target=train, args=(rank, args, model))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    print("Finished training model")


if __name__ == "__main__":
    main()
