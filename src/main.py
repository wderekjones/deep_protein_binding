import torch
from tqdm import tqdm
import argparse
import numpy as np
from multiprocessing import cpu_count
from MoleculeDataset import MoleculeDatasetH5
from torch.utils.data import DataLoader

from model import MPNN
from label import read_labels
from loss import MSELoss

parser = argparse.ArgumentParser()
parser.add_argument('-D', type=str, help="path to dataset")
parser.add_argument('-L', type=str, help="path to labels to use")
parser.add_argument("--nworkers", type=int, help="number of workers to use in data loader", default=1)
parser.add_argument("--batch_size", type=int, help="batch size to use in data loader", default=1)

args = parser.parse_args()


if __name__ == "__main__":

    # labels = read_labels("/u/vul-d1/scratch/wdjo224/data/Informative_features.csv", "/media/derek/Data/thesis_data/null_column_list.csv")


    model = MPNN(3)

    # model.cuda().half()
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.SmoothL1Loss()
    print("training model")

    data = MoleculeDatasetH5(data_dir="/mounts/u-vul-d1/scratch/wdjo224/data/deep_protein_binding/datasets",
                             list_dir="/mounts/u-vul-d1/scratch/wdjo224/data/deep_protein_binding/dataset_compounds",
                             corrupt_path="/u/vul-d1/scratch/wdjo224/data/deep_protein_binding/corrupt_inputs.csv",
                             targets=["label"], num_workers=1)
    print("size of dataset: {}".format(len(data)))
    from torch.utils.data import DataLoader


    def collate_fn(batch):

        return batch

    epochs = 2
    batch_size = args.batch_size
    num_workers = args.nworkers
    num_iters = int(np.ceil(len(data) / batch_size))
    mydata = DataLoader(data, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)
    print("batch size: {} \t num_iterations: {} \t num_workers: {}".format(batch_size, num_iters, num_workers))
    for epoch in range(0,epochs):
        for idx, batch in tqdm(enumerate(mydata), total=num_iters):
            # just here to take up space
            data = batch[0]
            y_pred = model(h=data["h"], g=data["g"])
            y_true = batch[0]["target"]
            loss = loss_fn(y_pred, torch.autograd.Variable(torch.FloatTensor(y_true)))
            print("loss: {}".format(loss.data.numpy()))
            loss.backward()
        optimizer.step()
    # for epoch in range(num_epochs):
    #     epoch_losses = []
    #     optimizer.zero_grad()
    #     for i_batch, sample_batched in tqdm(enumerate(molecule_loader), total=batch_size, leave=True):
    #         pred = model(torch.autograd.Variable(sample_batched['atom'].type(torch.HalfTensor).cuda()),
    #         torch.autograd.Variable(sample_batched['bond'].type(torch.HalfTensor).cuda()),
    #         torch.autograd.Variable(sample_batched['edge'].type(torch.HalfTensor).cuda()))
    #         loss = loss_fn(pred.float(), torch.autograd.Variable(sample_batched['target'].cuda().float()))
    #         loss.backward()
    #         epoch_losses.append(loss.data.cpu().numpy()) # is this a bottleneck?
    #     optimizer.step()
    #     print("epoch: {} \t loss: {} ".format(epoch, np.mean(epoch_losses)))
