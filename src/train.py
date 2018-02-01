

if __name__ == "__main__":
    import torch
    from tqdm import tqdm
    import argparse
    import numpy as np
    from MoleculeDataset import MoleculeDatasetH5, MoleculeDatasetCSV
    from torch.utils.data import DataLoader
    from sklearn.metrics import r2_score, f1_score
    from model import MPNN
    from loss import MSELoss
    from utils import collate_fn




    parser = argparse.ArgumentParser()
    parser.add_argument('-D', type=str, help="path to dataset")
    parser.add_argument('-L', type=str, help="path to labels to use")
    parser.add_argument("--n_workers", type=int, help="number of workers to use in data loader", default=0)
    parser.add_argument("--batch_size", type=int, help="batch size to use in data loader", default=1)
    parser.add_argument("--n_epochs", type=int, help="number of training epochs", default=1)
    parser.add_argument("--use_cuda", type=bool, help="indicates whether to use gpu", default=False)
    parser.add_argument("--lr", type=float, help="learning rate to use for training", default=1e-3)
    args = parser.parse_args()

    import torch.multiprocessing as mp
    mp = mp.get_context("forkserver")

    print("{:=^100}".format(' Testing SGD '))

    model = MPNN(3)

    if args.use_cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()
    print("training model")

    # data = MoleculeDatasetH5(data_dir="/mounts/u-vul-d1/scratch/wdjo224/data/deep_protein_binding/datasets", list_dir="/mounts/u-vul-d1/scratch/wdjo224/data/deep_protein_binding/dataset_compounds",
    #                              corrupt_path="/u/vul-d1/scratch/wdjo224/data/deep_protein_binding/corrupt_inputs.csv",targets=["label"],num_workers=1)

    molecules = MoleculeDatasetCSV(csv_file="/u/vul-d1/scratch/wdjo224/data/deep_protein_binding/kinase_no_duplicates_with_smiles.csv",
                              corrupt_path="/u/vul-d1/scratch/wdjo224/data/deep_protein_binding/corrupt_inputs.csv", targets=["MLOGP"],cuda=args.use_cuda)

    # TODO: may need to preprocess targets to ensure that no NANs, inf, etc; values are present

    print("size of dataset: {}".format(len(molecules)))
    from torch.utils.data import DataLoader

    epochs = args.n_epochs
    batch_size = args.batch_size
    num_workers = args.n_workers
    num_iters = int(np.ceil(len(molecules) / batch_size))
    molecule_loader = DataLoader(molecules, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)
    print("batch size: {} \t learning rate: {} \t num_iterations: {} \t num_workers: {}".format(batch_size, args.lr, num_iters, num_workers))
    for epoch in range(0, epochs):
        epoch_loss = []
        epoch_preds = []
        epoch_true = []

        # for idx, batch in tqdm(enumerate(molecule_loader), total=num_iters):
        for idx, batch in enumerate(molecule_loader):
            optimizer.zero_grad()
            data = batch[0]
            y_pred = model(h=data["h"], g=data["g"])
            y_true = batch[0]["target"]
            # loss = loss_fn(y_pred, torch.autograd.Variable(torch.cuda.FloatTensor(y_true)))
            loss = loss_fn(y_pred, torch.autograd.Variable(torch.FloatTensor(y_true)))
            loss.backward()
            # epoch_loss.append(loss.cpu().data.numpy())
            epoch_loss.append(loss.data.numpy())
            # epoch_preds.append(y_pred.cpu().data.numpy())
            epoch_preds.append(y_pred.data.numpy()[0])
            epoch_true.append(y_true)
            optimizer.step()
            print("epoch: {} \t \t loss: {} \t r2_score: {}".format(epoch, np.mean(epoch_loss),
                                                                         r2_score(epoch_true, epoch_preds)))

        # optimizer.step()
