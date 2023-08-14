from torch_geometric.loader import DataLoader
from pytorch_lightning import LightningDataModule
import torch

from musgconv.data.datasets import  Bach370ChoralesPGVoiceSeparationDataset


class GraphPGDataModule(LightningDataModule):
    def __init__(self, batch_size=1, num_workers=4, force_reload=False):
        super(GraphPGDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.force_reload = force_reload

    def prepare_data(self):
        Bach370ChoralesPGVoiceSeparationDataset(force_reload=self.force_reload, nprocs=self.num_workers)

    def setup(self, stage=None):
        dataset = Bach370ChoralesPGVoiceSeparationDataset(nprocs=self.num_workers)
        self.metadata = dataset.metadata
        idx = torch.randperm(len(dataset)).long()
        nidx = int(len(idx) * 0.7)
        self.train_idx = idx[:nidx]
        self.val_idx = idx[nidx:]
        self.dataset_train = dataset[self.train_idx]
        self.dataset_val = dataset[self.val_idx]
        self.dataset_predict = dataset[self.val_idx[:5]]
        self.features = dataset.features

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return DataLoader(self.dataset_predict, batch_size=self.batch_size, num_workers=self.num_workers)