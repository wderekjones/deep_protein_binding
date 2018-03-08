import os
import torch
from utils import get_args, update_tensorboard
from multiprocessing import Value
from multiprocessing.managers import BaseManager
from tensorboardX import SummaryWriter

args = get_args()

root = "/scratch/wdjo224/deep_protein_binding/"
experiment_path = root + "experiments/" + args.exp_name + "/" + args.exp_name
checkpoint_path = experiment_path + "/checkpoints/"
log_path = root + "logs/"
result_path = experiment_path + "/results/"
scalar_path = experiment_path + "/results/" + args.exp_name + "_all_scalars.json"


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


def train_epoch(rank, epoch, global_step, model, molecule_loader_train, molecule_loader_val, optimizer, loss_fn, writer):
    torch.manual_seed(args.seed+rank)
    model.train()
    pid = os.getpid()

    for batch_idx, batch in enumerate(molecule_loader_train):

        # zero the gradients for the next batch
        optimizer.zero_grad()

        # take a training step, i.e. process the mini-batch and accumulate gradients
        train_dict = model.train_step(batch=batch, loss_fn=loss_fn)


        # log the information to tensorboard
        update_tensorboard(writer=writer, train_dict=train_dict, val_dict=None, step=global_step.value)

        print("pid: {} \t epoch: {} \t step: {} \t train loss: {}".format(pid, epoch, global_step.value,
                                                                          train_dict["loss"].data))

        if batch_idx % 10 == 0:

            # take a validation step for every 10 training steps
            val_dict = model.validation_step(batch=next(iter(molecule_loader_val)), loss_fn=loss_fn)

            print("\n pid: {} \t epoch: {} \t step: {} \t val loss: {}".format(pid, epoch, batch_idx,
                                                                               val_dict["loss"].data))

            # log the information to tensorboard
            update_tensorboard(writer=writer, train_dict=train_dict, val_dict=val_dict, step=global_step.value)

        # update the model parameters
        optimizer.step()

        # increment the global step variable
        global_step.value += 1


class TFWriterManager(BaseManager):
    pass

TFWriterManager.register('SummaryWriter', SummaryWriter)

def main():

    import torch.multiprocessing as mp
    mp.set_sharing_strategy("file_system")
    import sys
    from torch.utils.data.sampler import SubsetRandomSampler
    import numpy as np
    from MoleculeDataset import MoleculeDatasetCSV
    from torch.utils.data import DataLoader
    from model import MPNN
    from utils import collate_fn, get_loss

    print("{:=^100}".format(' Train '))
    print("experiment: {}".format(args.exp_name))
    print("run parameters: {} \n".format(sys.argv))

    global_step = Value('i',0)


    manager = TFWriterManager()
    manager.start()
    writer = manager.SummaryWriter(log_path + args.exp_name)

    writer.add_text('args', str(sys.argv))
    writer.add_text("target", str(args.target))

    print("loading data...")

    molecules = MoleculeDatasetCSV(csv_file=args.D, corrupt_path=args.c, target=args.target, cuda=args.use_cuda,
                                   scaling=args.scale)

    model_path = args.model_path
    epochs = args.n_epochs
    batch_size = args.batch_size
    train_idxs = np.fromfile(args.train_idxs, dtype=np.int)
    val_idxs = np.fromfile(args.val_idxs, dtype=np.int)

    molecule_loader_train = DataLoader(molecules, batch_size=batch_size, num_workers=args.n_workers, collate_fn=collate_fn,
                                       sampler=SubsetRandomSampler(train_idxs))
    molecule_loader_val = DataLoader(molecules, batch_size=batch_size, num_workers=args.n_workers, collate_fn=collate_fn,
                                       sampler=SubsetRandomSampler(val_idxs))

    print("instantiating model...")
    model = MPNN(T=args.T, p=args.p, target=args.target, output_type=args.output_type, output_dim=args.output_dim,
                 readout_dim=args.readout_dim)
    if model_path is not None:
        model.load_state_dict(model_path)

    model.share_memory()

    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_fn = get_loss(args)

    # Train the model
    print("Training Model...")

    for epoch in range(0, epochs):
        processes = []
        for rank in range(args.n_train_process):
            p = mp.Process(target=train_epoch, args=(rank, epoch, global_step, model, molecule_loader_train,
                                                     molecule_loader_val, optimizer, loss_fn, writer))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        print("Saving model checkpoint...")
        torch.save(model.state_dict(), checkpoint_path + args.exp_name + "_epoch" + str(epoch))

    print("Finished training model")

    # Output training metrics
    print("Saving training metrics")

    writer.export_scalars_to_json(scalar_path)
    writer.close()


if __name__ == "__main__":
    main()
