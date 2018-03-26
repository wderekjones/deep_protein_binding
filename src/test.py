import sys
sys.path.append("/scratch/wdjo224/deep_protein_binding")
import torch
torch.manual_seed(0)
import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from itertools import chain
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from src.model import MPNN
from src.MoleculeDataset import MoleculeDatasetCSV
from src.utils import collate_fn, get_loss, get_parser

#TODO: output feature vectors

args = get_parser().parse_args()

test_idxs = np.array_split(np.fromfile(args.val_idxs, dtype=np.int), args.n_test_process)

output_path = "/scratch/wdjo224/deep_protein_binding/experiments/" + args.exp_name + "/test_results/{}".format(args.pid)


def test(rank, model):
    idxs = test_idxs[rank]
    print("pid: {}".format(os.getpid()))
    result_summary = pd.DataFrame({"idx": [], "pred": [], "true": [], "loss": []})

    molecules = MoleculeDatasetCSV(
        csv_file=args.D,
        corrupt_path=args.c, target=args.target, scaling=args.scale)

    loss_fn = get_loss(args)

    start_time = time.clock()
    for idx in idxs:
        molecule_loader = DataLoader(molecules, batch_size=1, sampler=SubsetRandomSampler([idx]),
                                     collate_fn=collate_fn, num_workers=0)
        for batch in molecule_loader:
            val_dict = model.validation_step(batch=batch, loss_fn=loss_fn)

            result_summary = pd.concat([result_summary, pd.DataFrame({"idx": [idx],
                                                                "loss": [val_dict["batch_dict"][key]["loss"]],
                                                                "pred": [val_dict["batch_dict"][key]["pred"]],
                                                                "true": [val_dict["batch_dict"][key]["true"]]}
                                                                for key in val_dict["batch_dict"].keys())], axis=0)
    end_time = time.clock()

    print("evaluation finished in {} cpu seconds. writing results...".format((end_time-start_time)))

    # convert the pred and true columns to numpy objects...have some messy shapes/etc so clean this up here
    result_summary.idx = result_summary.idx.apply(lambda x: x[0])
    result_summary.pred = result_summary.pred.apply(lambda x: [x[0][0][0].data.numpy()[0], x[0][0][1].data.numpy()[0]])
    result_summary.true = result_summary.true.apply(lambda x: [x[0][0][0].data.numpy()[0], x[0][0][1].data.numpy()[0]])
    result_summary.loss = result_summary.loss.apply(lambda x: x[0][0])

    result_summary = result_summary.reset_index()

    result_summary.to_csv(output_path+"/test_results_{}.csv".format(rank))


def main():

    print("{:=^100}".format(' Test '))
    print("run parameters: {}".format(sys.argv))

    import torch.multiprocessing as mp
    mp.set_sharing_strategy("file_system")

    # if output path does not exist, create it
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model = MPNN(T=args.T, p=args.p, target=args.target, output_type=args.output_type, output_dim=args.output_dim,
                 readout_dim=args.readout_dim)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    processes = []
    for rank in range(args.n_test_process):
        p = mp.Process(target=test, args=(rank, model))
        p.start()
        processes.append(p)

    print("joining {} processes.".format(len(processes)))

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()


