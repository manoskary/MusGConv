from torch_geometric.loader import DataLoader as PygDataLoader
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import DataLoader
from musgconv.data.datasets import MCMAGraphPGVoiceSeparationDataset, MCMAGraphVoiceSeparationDataset
from torch.nn import functional as F


class GraphPGMCMADataModule(LightningDataModule):
    def __init__(self, batch_size=1, num_workers=4, force_reload=False, test_collection=None):
        super(GraphPGMCMADataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.force_reload = force_reload
        self.dataset = MCMAGraphPGVoiceSeparationDataset(force_reload=self.force_reload, nprocs=self.num_workers)
        self.features = self.dataset.features
        self.test_collection = test_collection

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.metadata = self.dataset.metadata
        if self.test_collection is None:
            idx = torch.randperm(len(self.dataset)).long()
            nidx = int(len(idx) * 0.7)
            self.train_idx = idx[:nidx]
            self.val_idx = idx[nidx:]
            self.dataset_train = self.dataset[self.train_idx]
            self.dataset_val = self.dataset[self.val_idx]
            self.dataset_predict = self.dataset[self.val_idx[:5]]
            print(f"Train size :{len(self.dataset_train)}, Val size :{len(self.dataset_val)}")
        else:
            idxs = torch.randperm(len(self.dataset)).long()
            # test_idx = idxs[self.dataset.graphs[idx].collection == self.test_collection]
            test_idx = [i for i in idxs if self.dataset.graphs[i].collection == self.test_collection]
            trainval_idx = [i for i in idxs if i not in test_idx]
            nidx = int(len(trainval_idx) * 0.9)
            train_idx = trainval_idx[:nidx]
            val_idx = trainval_idx[nidx:]

            self.dataset_train = self.dataset[train_idx]
            self.dataset_val = self.dataset[val_idx]
            self.dataset_test = self.dataset[test_idx]
            self.dataset_predict = self.dataset[test_idx[:5]]
            print(f"Running evaluation on collection {self.test_collection}")
            print(f"Train size :{len(self.dataset_train)}, Val size :{len(self.dataset_val)}, Test size :{len(self.dataset_test)}")
            

    def train_dataloader(self):
        return PygDataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return PygDataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self) :
        return PygDataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers)

    def predict_dataloader(self):
        return PygDataLoader(self.dataset_predict, batch_size=self.batch_size, num_workers=self.num_workers)


class MCMAGraphDataModule(LightningDataModule):
    """

    """
    def __init__(self, batch_size=1, num_workers=4, force_reload=False, test_collection=None, normalize_features=True):
        self.test_collection = test_collection
        super(MCMAGraphDataModule, self).__init__()
        self.batch_size = batch_size
        self.normalize_features = normalize_features
        self.num_workers = num_workers
        self.force_reload = force_reload
        self.dataset = MCMAGraphVoiceSeparationDataset(force_reload=self.force_reload, nprocs=self.num_workers)
        self.features = self.dataset.features

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        # self.metadata = self.dataset.metadata
        if self.test_collection is None:
            idx = torch.randperm(len(self.dataset)).long()
            nidx = int(len(idx) * 0.7)
            self.train_idx = idx[:nidx]
            self.val_idx = idx[nidx:]
            self.dataset_train = self.dataset[self.train_idx]
            self.dataset_val = self.dataset[self.val_idx]
            self.dataset_predict = self.dataset[self.val_idx[:5]]
        else:
            idxs = torch.randperm(len(self.dataset)).long()
            # test_idx = idxs[self.dataset.graphs[idx].collection == self.test_collection]
            test_idx = [i for i in idxs if self.dataset.graphs[i].collection == self.test_collection]
            trainval_idx = [i for i in idxs if i not in test_idx]
            nidx = int(len(trainval_idx) * 0.9)
            train_idx = trainval_idx[:nidx]
            val_idx = trainval_idx[nidx:]

            self.dataset_train = self.dataset[train_idx]
            self.dataset_val = self.dataset[val_idx]
            self.dataset_test = self.dataset[test_idx]
            self.dataset_predict = self.dataset[test_idx[:5]]

    def collate_fn(self, batch):
        batch_inputs, edges, batch_label, edge_type, pot_edges, truth_edges, na, name = batch[0]
        batch_inputs = F.normalize(batch_inputs.squeeze(0)) if self.normalize_features else batch_inputs.squeeze(0)
        batch_label = batch_label.squeeze(0)
        edges = edges.squeeze(0)
        edge_type = edge_type.squeeze(0)
        pot_edges = pot_edges.squeeze(0)
        truth_edges = torch.tensor(truth_edges.squeeze()).to(pot_edges.device)
        na = torch.tensor(na)
        return batch_inputs, edges, batch_label, edge_type, pot_edges, truth_edges, na, name[0]

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.dataset_predict, batch_size=self.batch_size, num_workers=self.num_workers)