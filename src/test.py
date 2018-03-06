'''

    script that loads a trained model and evaluates on specified testing data
    by: Derek Jones

'''

#TODO: output predictions
#TODO: output feature vectors

from utils import get_args
args = get_args()

if __name__ == "__main__":
    import torch
    torch.manual_seed(0)
    import sys
    import os
    import pandas as pd
    import numpy as np
    from torch.utils.data.sampler import SubsetRandomSampler
    from torch.utils.data import DataLoader
    from model import MPNN
    from MoleculeDataset import MoleculeDatasetCSV
    from utils import collate_fn, get_loss

    print("{:=^100}".format(' Test '))
    print("run parameters: {}".format(sys.argv))

    # TODO: use a multiindex instead
    output_r2_summary = pd.DataFrame({key: [] for key in args.target_list})
    output_loss_summary = pd.DataFrame({key: [] for key in args.target_list})
    model = MPNN(T=args.T, p=args.p, target_list=args.target_list, output_type=args.output_type, output_dim=args.output_dim)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    molecules = MoleculeDatasetCSV(
        csv_file=args.D,
        corrupt_path=args.c, targets=args.target_list,
        cuda=args.use_cuda, scaling=args.scale)

    test_idxs = np.fromfile(args.test_idxs, dtype=np.int)
    molecule_loader = DataLoader(molecules, batch_size=args.batch_size, sampler=SubsetRandomSampler(test_idxs),
                                 collate_fn=collate_fn, num_workers=args.n_workers)
    loss_fn = get_loss(args)

    if args.use_cuda:
        loss_fn.cuda()

    hidden_list = []
    #TODO: implement multiprocessing to process n batches in parallel?
    for idx, batch in enumerate(molecule_loader):
        val_dict = model.validation_step(batch=batch, loss_fn=loss_fn)

        if args.output_type == "regress":
            output_r2_summary = pd.concat([output_r2_summary, pd.DataFrame({key: [val_dict["batch_dict"][key]["r2"]]
                                                                  for key in val_dict["batch_dict"].keys()})], axis=0)

            output_loss_summary = pd.concat([output_loss_summary, pd.DataFrame({key: val_dict["batch_dict"][key]["loss"]
                                                                  for key in val_dict["batch_dict"].keys()})], axis=0)
        elif args.output_type == "class":
            output_r2_summary = pd.concat([output_r2_summary, pd.DataFrame([val_dict["batch_dict"][key]["metrics"]
                                                                for key in val_dict["batch_dict"].keys()])],axis=0)

            output_loss_summary = pd.concat([output_loss_summary, pd.DataFrame([val_dict["batch_dict"][key]["loss"]
                                                                                           for key in val_dict["batch_dict"].keys()])]
                                            , axis=0)
        print("\nstep: {} \t val loss: {}".format(idx, val_dict["loss"].data))

    print("saving results...")
    if not os.path.exists("results/"):
        os.makedirs("results/")

    output_loss_summary.to_csv("results/"+args.exp_name+"_test_loss.csv", index=False)
    output_r2_summary.to_csv("results/"+args.exp_name+"_test_r2.csv", index=False)

    print("evaluation complete.")
