from utils import get_args
args = get_args()

if __name__ == "__main__":
    import torch.multiprocessing as mp
    # mp.set_start_method("spawn")
    mp.set_sharing_strategy("file_system")
    mp = mp.get_context("spawn")

    import os
    import time
    import sys
    import torch
    torch.manual_seed(0)
    from torch.utils.data.sampler import SubsetRandomSampler
    import numpy as np
    from MoleculeDataset import MoleculeDatasetH5, MoleculeDatasetCSV
    from torch.utils.data import DataLoader
    from tensorboardX import SummaryWriter
    from model import MPNN
    from utils import collate_fn, update_tensorboard, get_loss

    print("{:=^100}".format(' Train '))
    print("experiment: {}".format(args.exp_name))
    print("run parameters: {} \n".format(sys.argv))
    root = "/scratch/wdjo224/deep_protein_binding/"
    if not os.path.exists(root+"experiments/"+args.exp_name):
        os.makedirs(root+"experiments/"+args.exp_name)
    exp_time = str(time.time())
    experiment_path = root+"experiments/"+args.exp_name + "/" + args.exp_name+"_"+exp_time

    print("loading data...")

    molecules = MoleculeDatasetCSV(csv_file=args.D,
                              corrupt_path=args.c, target=args.target, cuda=args.use_cuda, scaling=args.scale)

    model_path = args.model_path
    epochs = args.n_epochs
    batch_size = args.batch_size
    num_workers = args.n_workers
    train_idxs = np.fromfile(args.train_idxs, dtype=np.int)
    val_idxs = np.fromfile(args.val_idxs, dtype=np.int)
    num_iters = int(np.ceil(train_idxs.shape[0] / batch_size))

    molecule_loader_train = DataLoader(molecules, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn,
                                       sampler=SubsetRandomSampler(train_idxs))
    molecule_loader_val = DataLoader(molecules, batch_size=batch_size, num_workers=0, collate_fn=collate_fn,
                                       sampler=SubsetRandomSampler(val_idxs))

    print("instantiating model...")
    model = MPNN(T=args.T, p=args.p, target=args.target, output_type=args.output_type, output_dim=args.output_dim,
                 readout_dim=args.readout_dim)
    if model_path is not None:
        model.load_state_dict(model_path)

    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_fn = get_loss(args)

    print("initializing tensorboard writer...")
    # Create a writer for tensorboard, add the command line arguments and targets to tensorboard
    if not os.path.exists(root+"logs/"):
        os.makedirs(root+"logs/")
    writer = SummaryWriter(root+"logs/"+args.exp_name+"_"+str(exp_time))
    writer.add_text('args', str(sys.argv))
    writer.add_text("target", str(args.target))


    global_step = 0

    # Train the model
    print("Training Model...")
    for epoch in range(0, epochs):

        for idx, batch in enumerate(molecule_loader_train):

            # zero the gradients for the next batch
            optimizer.zero_grad()

            # take a training step, i.e. process the mini-batch and accumulate gradients
            train_dict = model.train_step(batch=batch, loss_fn=loss_fn)

            print("epoch: {} \t step: {} \t train loss: {}".format(epoch, idx,
                                                                             train_dict["loss"].data))

            if idx % 10 == 0:

                # take a validation step for every 10 training steps
                val_dict = model.validation_step(batch=next(iter(molecule_loader_val)), loss_fn=loss_fn)

                print("\n epoch: {} \t step: {} \t val loss: {}".format(epoch, idx,
                                                                        val_dict["loss"].data))
                # log the information to tensorboard
                update_tensorboard(writer=writer, train_dict=train_dict, val_dict=val_dict, step=global_step)
            else:
                # log the information to tensorboard

                update_tensorboard(writer=writer, train_dict=train_dict, val_dict=None, step=global_step)

            if idx % 100 == 0:
                print("Saving model checkpoint...")
                if not os.path.exists(experiment_path + "/checkpoints/"):
                    os.makedirs(experiment_path + "/checkpoints/")
                torch.save(model.state_dict(),
                           experiment_path + "/checkpoints/" + args.exp_name + "_" + str(exp_time) + "_epoch" + str(
                               epoch)+"_step_"+str(global_step))
            # update the model parameters
            optimizer.step()

            global_step += 1

    print("Finished training model")

    # Output training metrics
    print("Saving training metrics")
    scalar_path = experiment_path+"/results/" + args.exp_name+"_"+str(exp_time)+ "_all_scalars.json"
    if not os.path.exists(experiment_path+"/results/"):
        os.makedirs(experiment_path+"/results/")

    writer.export_scalars_to_json(scalar_path)
    writer.close()

