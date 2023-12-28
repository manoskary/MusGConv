from torch_geometric.loader import DataLoader as PygDataLoader
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import ConcatDataset
from musgconv.data.datasets import (
    MozartStringQuartetCadenceGraphDataset,
    MozartPianoSonatasCadenceGraphDataset,
    HaydnStringQuartetsCadenceGraphDataset,
    BachWTCCadenceGraphDataset,
)
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from musgconv.data.datamodules.mix_vs import idx_tuple_to_dict
from musgconv.data.samplers import BySequenceLengthSampler
import numpy as np


class GraphCadenceDataModule(LightningDataModule):
    def __init__(
            self, batch_size=50, num_workers=4, force_reload=False, verbose=False, include_measures=False, max_size=10000, use_all_features=False
    ):
        super(GraphCadenceDataModule, self).__init__()
        self.batch_size = batch_size
        self.bucket_boundaries = [200, 300, 400, 500, 700, 1000]
        self.num_workers = num_workers
        self.force_reload = force_reload
        self.normalize_features = True
        self.include_measures = include_measures
        self.use_all_features = use_all_features
        self.datasets = [
            MozartStringQuartetCadenceGraphDataset(force_reload=self.force_reload, nprocs=self.num_workers, verbose=verbose, max_size=max_size),
            MozartPianoSonatasCadenceGraphDataset(force_reload=self.force_reload, nprocs=self.num_workers, verbose=verbose, max_size=max_size),
            HaydnStringQuartetsCadenceGraphDataset(force_reload=self.force_reload, nprocs=self.num_workers, verbose=verbose, max_size=max_size),
            BachWTCCadenceGraphDataset(force_reload=self.force_reload, nprocs=self.num_workers, verbose=verbose, max_size=max_size),
        ]
        if not (all([d.features == self.datasets[0].features for d in self.datasets])):
            raise Exception("Input dataset has different features, Datasets {} with sizes: {}".format(
                " ".join([d.name for d in self.datasets]), " ".join([str(d.features) for d in self.datasets])))
        # all datasets have the same features
        self.features = self.datasets[0].features if self.use_all_features else 16

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.datasets_map = [(dataset_i, piece_i) for dataset_i, dataset in enumerate(self.datasets) for piece_i in
                             range(len(dataset))]

        idxs = range(len(self.datasets_map))

        trainval_idx, test_idx = train_test_split(idxs, test_size=0.2, random_state=0)
        train_idx, val_idx = train_test_split(trainval_idx, test_size=0.1, random_state=0)

        # structure the indices as dicts {dataset_i : [piece_i,...,piece_i]}
        self.test_idx_dict = idx_tuple_to_dict(test_idx, self.datasets_map)
        self.train_idx_dict = idx_tuple_to_dict(train_idx, self.datasets_map)
        self.val_idx_dict = idx_tuple_to_dict(val_idx, self.datasets_map)


        print(
            f"Train size :{len(train_idx)}, Val size :{len(val_idx)}, Test size :{len(test_idx)}"
        )

    def collate_fn(self, batch):
        out = {}
        e = batch[0]
        if self.include_measures:
            out["beat_nodes"] = e["beat_nodes"].long().squeeze()
            out["beat_edges"] = e["beat_edges"].long().squeeze()
            out["measure_nodes"] = e["measure_nodes"].long().squeeze()
            out["measure_edges"] = e["measure_edges"].long().squeeze()
        out["x"] = e["x"].squeeze(0) if self.use_all_features else e["x"][:, :self.features]
        out["y"] = e["y"].squeeze(0)
        out["edge_index"] = e["edge_index"].squeeze(0)
        out["edge_type"] = e["edge_type"].squeeze(0)
        out["note_array"] = torch.tensor(e["note_array"]).float()
        return out

    def collate_train_fn(self, examples):
        out = {}
        lengths = list()
        x = list()
        edge_index = list()
        edge_types = list()
        y = list()
        note_array = list()
        potential_edges = list()
        true_edges = list()
        max_idx = []
        beats = []
        beat_eindex = []
        measures = []
        measure_eindex = []
        for e in examples:
            if self.include_measures:
                beats.append(e["beat_nodes"].long())
                beat_eindex.append(e["beat_edges"].long())
                measures.append(e["measure_nodes"].long())
                measure_eindex.append(e["measure_edges"].long())
            x.append(e["x"] if self.use_all_features else e["x"][:, :self.features])
            lengths.append(e["x"].shape[0])
            edge_index.append(e["edge_index"])
            edge_types.append(e["edge_type"])
            y.append(e["y"])
            note_array.append(torch.tensor(e["note_array"]))
            max_idx.append(e["x"].shape[0])
        lengths = torch.tensor(lengths).long()
        out["lengths"], perm_idx = lengths.sort(descending=True)
        perm_idx = perm_idx.tolist()
        max_idx = np.cumsum(np.array([0] + [max_idx[i] for i in perm_idx]))
        out["x"] = torch.cat([x[i] for i in perm_idx], dim=0).float()
        out["edge_index"] = torch.cat([edge_index[pi]+max_idx[i] for i, pi in enumerate(perm_idx)], dim=1).long()
        out["edge_type"] = torch.cat([edge_types[i] for i in perm_idx], dim=0).long()
        out["y"] = torch.cat([y[i] for i in perm_idx], dim=0).long()
        out["note_array"] = torch.cat([note_array[i] for i in perm_idx], dim=0).float()
        if self.include_measures:
            max_beat_idx = np.cumsum(np.array([0] + [beats[i].shape[0] for i in perm_idx]))
            out["beat_nodes"] = torch.cat([beats[pi] + max_beat_idx[i] for i, pi in enumerate(perm_idx)], dim=0).long()
            out["beat_lengths"] = torch.tensor(max_beat_idx).long()
            out["beat_edges"] = torch.cat([torch.vstack((beat_eindex[pi][0] + max_idx[i], beat_eindex[pi][1] + max_beat_idx[i])) for i, pi in enumerate(perm_idx)], dim=1).long()
            max_measure_idx = np.cumsum(np.array([0] + [measures[i].shape[0] for i in perm_idx]))
            out["measure_nodes"] = torch.cat([measures[pi] + max_measure_idx[i] for i, pi in enumerate(perm_idx)], dim=0).long()
            out["measure_edges"] = torch.cat([torch.vstack((measure_eindex[pi][0] + max_idx[i], measure_eindex[pi][1] + max_measure_idx[i])) for i, pi in enumerate(perm_idx)], dim=1).long()
            out["measure_lengths"] = torch.tensor(max_measure_idx).long()
        return out

    def train_dataloader(self):
        self.dataset_train = ConcatDataset([self.datasets[k][self.train_idx_dict[k]] for k in self.train_idx_dict.keys()])
        sampler = BySequenceLengthSampler(self.dataset_train, self.bucket_boundaries, self.batch_size)
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_sampler=sampler,
            batch_size=1,
            num_workers=0,
            collate_fn=self.collate_train_fn,
            drop_last=False,
            pin_memory=False,
        )

    def val_dataloader(self):
        self.dataset_val = ConcatDataset([self.datasets[k][self.val_idx_dict[k]] for k in self.val_idx_dict.keys()])
        return torch.utils.data.DataLoader(
            self.dataset_val, batch_size=1, num_workers=self.num_workers, collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        self.dataset_test = ConcatDataset([self.datasets[k][self.test_idx_dict[k]] for k in self.test_idx_dict.keys()])
        return torch.utils.data.DataLoader(
            self.dataset_test, batch_size=1, num_workers=self.num_workers, collate_fn=self.collate_fn
        )