import os
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from torch.utils.data import DataLoader
from utils import get_args,get_opt, update_tensorboard
from multiprocessing import Value
from multiprocessing.managers import BaseManager
from MoleculeDataset import MoleculeDatasetCSV
from tensorboardX import SummaryWriter
from utils import collate_fn, get_loss

args = get_args()

root = "/scratch/wdjo224/deep_protein_binding/"
experiment_path = root + "experiments/" + args.exp_name + "/" + args.exp_name
checkpoint_path = experiment_path + "/checkpoints/"
log_path = root + "logs/"
result_path = experiment_path + "/results/"
scalar_path = experiment_path + "/results/" + args.exp_name + "_all_scalars.json"

# create a global shared counter, init to -1 so that first process becomes 0th step and so on...
global_step = Value('i', -1)

# create a driectory to store experiment data
if not os.path.exists(root + "experiments/" + args.exp_name):
    print("creating experiment path...")
    os.makedirs(root + "experiments/" + args.exp_name)

# create a directory to store checkpoints
if not os.path.exists(checkpoint_path):
    print("creating checkpoint path...")
    os.makedirs(checkpoint_path)

if not os.path.exists(result_path):
    print("creating result path...")
    os.makedirs(result_path)

# Create a writer for tensorboard, add the command line arguments and targets to tensorboard
if not os.path.exists(log_path):
    print("creating tensorboard log path...")
    os.makedirs(log_path)


def train(rank, args, model, writer):

    torch.manual_seed(args.seed + rank)

    print("loading data...")

    molecules = MoleculeDatasetCSV(csv_file=args.D, corrupt_path=args.c, target=args.target, cuda=args.use_cuda,
                                   scaling=args.scale)

    batch_size = args.batch_size
    train_idxs = np.fromfile(args.train_idxs, dtype=np.int)
    val_idxs = np.fromfile(args.val_idxs, dtype=np.int)

    molecule_loader_train = DataLoader(molecules, batch_size=batch_size, num_workers=args.n_workers,
                                       collate_fn=collate_fn,
                                       sampler=SubsetRandomSampler(train_idxs))
    # molecule_loader_val = DataLoader(molecules, batch_size=batch_size, num_workers=args.n_workers,
    #                                  collate_fn=collate_fn,
    #                                  sampler=SubsetRandomSampler(val_idxs))

    loss_fn = get_loss(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.n_epochs + 1):
        train_epoch(rank=rank, epoch=epoch, global_step=global_step, model=model,
                    molecule_loader_train=molecule_loader_train, optimizer=optimizer, loss_fn=loss_fn, writer=writer)
        print("Saving model checkpoint...")
        torch.save(model.state_dict(), checkpoint_path + args.exp_name + "_epoch" + str(epoch))


def train_epoch(rank, epoch, global_step, model, molecule_loader_train, optimizer, loss_fn, writer):
    torch.manual_seed(args.seed+rank)
    model.train()
    pid = os.getpid()

    for batch_idx, batch in enumerate(molecule_loader_train):

        # increment the global step variable
        global_step.value += 1

        # zero the gradients for the next batch
        optimizer.zero_grad()

        # take a training step, i.e. process the mini-batch and accumulate gradients
        train_dict = model.train_step(batch=batch, loss_fn=loss_fn)

        # log the information to tensorboard
        update_tensorboard(writer=writer, train_dict=train_dict, val_dict=None, step=global_step.value)

        print("pid: {} \t epoch: {} \t step: {} \t train loss: {}".format(pid, epoch, global_step.value,
                                                                          train_dict["loss"].data))

        # if global_step.value % 10 == 0:

            # take a validation step for every 10 training steps
            # val_dict = model.validation_step(batch=next(iter(molecule_loader_val)), loss_fn=loss_fn)

            # print("\n pid: {} \t epoch: {} \t step: {} \t val loss: {}".format(pid, epoch, global_step.value,
            #                                                                    val_dict["loss"].data))

            # log the information to tensorboard
            # update_tensorboard(writer=writer, train_dict=train_dict, val_dict=val_dict, step=global_step.value)

        # update the model parameters
        optimizer.step()


class TFWriterManager(BaseManager):
    pass


TFWriterManager.register('SummaryWriter', SummaryWriter)


def main():

    import torch.multiprocessing as mp
    mp.set_sharing_strategy("file_system")
    import sys
    from model import MPNN

    torch.manual_seed(args.seed)

    print("{:=^100}".format(' Train '))
    print("experiment: {}".format(args.exp_name))
    print("run parameters: {} \n".format(sys.argv))

    manager = TFWriterManager()
    manager.start()
    writer = manager.SummaryWriter(log_path + args.exp_name)

    writer.add_text('args', str(sys.argv))
    writer.add_text("target", str(args.target))

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
        p = mp.Process(target=train, args=(rank, args, model, writer))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    print("Finished training model")

    # Output training metrics
    print("Saving training metrics")

    writer.export_scalars_to_json(scalar_path)
    writer.close()


if __name__ == "__main__":
    main()
