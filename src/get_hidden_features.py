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
    from tqdm import tqdm
    import pandas as pd
    import numpy as np
    from torch.utils.data.sampler import SubsetRandomSampler
    from torch.utils.data import DataLoader
    from model import MPNN
    from MoleculeDataset import MoleculeDatasetCSV
    from utils import collate_fn, get_loss

    print("{:=^100}".format(' Test '))
    print("run parameters: {}".format(sys.argv))

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
    #TODO: store the labels, can access via batch[idx]["label"]
    hidden_feature_df = pd.DataFrame()
    label_df = pd.DataFrame()

    print("computing features")
    i = 0
    for batch in tqdm(molecule_loader):

        for example in batch:
            hidden_feature_df[i] = model(example["h"], example["g"], hidden_pass=True).data.numpy()
            label_df[i] = example["label"]
            i += 1
    print("saving hidden features")
    hidden_feature_df.transpose().to_csv(args.exp_name+"_hidden_features.csv", index=False)
    label_df.to_csv(args.exp_name+"_hidden_features_labels.csv", index=False)
    print("complete.")