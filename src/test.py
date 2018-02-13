'''

    script that loads a trained model and evaluates on specified testing data
    by: Derek Jones

'''

from utils import get_args
args = get_args()

if __name__ == "__main__":
    import torch
    torch.manual_seed(0)
    import sys
    import pandas as pd
    import numpy as np
    from torch.utils.data.sampler import SubsetRandomSampler
    from torch.utils.data import DataLoader
    from model import MPNN
    from MoleculeDataset import MoleculeDatasetCSV
    from utils import validation_step, collate_fn, get_loss

    print("{:=^100}".format(' Test '))
    print("run parameters: {}".format(sys.argv))

    # TODO: use a multiindex instead
    output_r2_summary = pd.DataFrame({key: [] for key in args.target_list})
    output_loss_summary = pd.DataFrame({key: [] for key in args.target_list})
    model = MPNN(T=args.T, p=args.p, n_tasks=len(args.target_list))
    model.load_state_dict(torch.load(args.model_path))
    model.eval()
    molecules = MoleculeDatasetCSV(
        csv_file="/u/vul-d1/scratch/wdjo224/data/deep_protein_binding/kinase_no_duplicates_with_smiles.csv",
        corrupt_path="/u/vul-d1/scratch/wdjo224/data/deep_protein_binding/corrupt_inputs.csv", targets=args.target_list,
        cuda=args.use_cuda, scaling=args.scale)

    test_idxs = np.fromfile(args.test_idxs, dtype=np.int)
    molecule_loader = DataLoader(molecules, batch_size=args.batch_size, sampler=SubsetRandomSampler(test_idxs),
                                 collate_fn=collate_fn, num_workers=args.n_workers)
    loss_fn = get_loss(args)

    if args.use_cuda:
        loss_fn.cuda()

    #TODO: implement multiprocessing to process n batches in parallel?
    for idx, batch in enumerate(molecule_loader):
        val_dict = validation_step(model=model, batch=batch, loss_fn=loss_fn,
                                   target_list=args.target_list, use_cuda=args.use_cuda)

        output_r2_summary = pd.concat([output_r2_summary, pd.DataFrame({key: [val_dict["target_dict"][key]["r2"]]
                                                                  for key in val_dict["target_dict"].keys()})], axis=0)

        output_loss_summary = pd.concat([output_loss_summary, pd.DataFrame({key: val_dict["target_dict"][key]["loss"]
                                                                  for key in val_dict["target_dict"].keys()})], axis=0)

        print("\nstep: {} \t val loss: {}".format(idx, val_dict["loss"].data))

    output_loss_summary.to_csv("results/"+args.exp_name+"_test_loss.csv", index=False)
    output_r2_summary.to_csv("results/"+args.exp_name+"_test_r2.csv", index=False)

