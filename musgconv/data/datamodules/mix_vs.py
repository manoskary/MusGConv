from torch_geometric.loader import DataLoader as PygDataLoader
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import ConcatDataset
from musgconv.data.datasets import (
    MCMAGraphPGVoiceSeparationDataset,
    Bach370ChoralesPGVoiceSeparationDataset,
    HaydnStringQuartetPGVoiceSeparationDataset,
    MCMAGraphVoiceSeparationDataset,
    Bach370ChoralesGraphVoiceSeparationDataset,
    HaydnStringQuartetGraphVoiceSeparationDataset,
    MozartStringQuartetGraphVoiceSeparationDataset,
    MozartStringQuartetPGGraphVoiceSeparationDataset,
    AugmentedNetChordGraphDataset,
    Augmented2022ChordGraphDataset,
)
from torch.nn import functional as F
from collections import defaultdict
from sklearn.model_selection import train_test_split
from musgconv.utils import add_reverse_edges_from_edge_index
from musgconv.data.samplers import BySequenceLengthSampler
import numpy as np


class GraphPGMixVSDataModule(LightningDataModule):
    def __init__(
        self, batch_size=1, num_workers=4, force_reload=False, test_collections=None, pot_edges_dist = 2
    ):
        super(GraphPGMixVSDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.force_reload = force_reload
        self.datasets = [
            # CrimGraphPGVoiceSeparationDataset(
            #     force_reload=self.force_reload, nprocs=self.num_workers, pot_edges_dist = pot_edges_dist
            # ),
            Bach370ChoralesPGVoiceSeparationDataset(
                force_reload=self.force_reload, nprocs=self.num_workers, pot_edges_dist = pot_edges_dist
            ),
            MCMAGraphPGVoiceSeparationDataset(
                force_reload=self.force_reload, nprocs=self.num_workers, pot_edges_dist=pot_edges_dist
            ),
            # HaydnStringQuartetPGVoiceSeparationDataset(
            #     force_reload=self.force_reload, nprocs=self.num_workers, pot_edges_dist=pot_edges_dist
            # ),
            # MozartStringQuartetPGGraphVoiceSeparationDataset(
            #     force_reload=self.force_reload, nprocs=self.num_workers, pot_edges_dist=pot_edges_dist
            # )

        ]
        if not (all([d.features == self.datasets[0].features for d in self.datasets])):
            raise Exception("Input dataset has different features")
        self.features = self.datasets[0].features
        self.test_collections = test_collections

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.datasets_map = [(dataset_i,piece_i) for dataset_i, dataset in enumerate(self.datasets) for piece_i in range(len(dataset))]
        if self.test_collections is None:
            idxs = range(len(self.datasets_map))
            collections = [self.datasets[self.datasets_map[i][0]].graphs[self.datasets_map[i][1]].collection for i in idxs]
            trainval_idx, test_idx = train_test_split(idxs, test_size=0.3, stratify=collections, random_state=0)
            trainval_collections = [collections[i] for i in trainval_idx]
            train_idx, val_idx = train_test_split(trainval_idx, test_size=0.1, stratify=trainval_collections, random_state=0)

            # structure the indices as dicts {dataset_i : [piece_i,...,piece_i]}
            test_idx_dict = idx_tuple_to_dict(test_idx, self.datasets_map)
            train_idx_dict = idx_tuple_to_dict(train_idx, self.datasets_map)
            val_idx_dict = idx_tuple_to_dict(val_idx, self.datasets_map)

            # create the datasets
            self.dataset_train = ConcatDataset([self.datasets[k][train_idx_dict[k]] for k in train_idx_dict.keys()])
            self.dataset_val = ConcatDataset([self.datasets[k][val_idx_dict[k]] for k in val_idx_dict.keys()])
            self.dataset_test = ConcatDataset([self.datasets[k][test_idx_dict[k]] for k in test_idx_dict.keys()])
            print("Running on all collections")
            print(
                f"Train size :{len(self.dataset_train)}, Val size :{len(self.dataset_val)}, Test size :{len(self.dataset_test)}"
            )
        else:
            # idxs = torch.randperm(len(self.datasets_map)).long()
            idxs = range(len(self.datasets_map))
            test_idx = [
                i
                for i in idxs
                if self.datasets[self.datasets_map[i][0]].graphs[self.datasets_map[i][1]].collection in self.test_collections
            ]
            trainval_idx = [i for i in idxs if i not in test_idx]
            trainval_collections = [self.datasets[self.datasets_map[i][0]].graphs[self.datasets_map[i][1]].collection for i in trainval_idx]
            train_idx, val_idx = train_test_split(trainval_idx, test_size=0.1, stratify=trainval_collections, random_state=0)
            # nidx = int(len(trainval_idx) * 0.9)
            # train_idx = trainval_idx[:nidx]
            # val_idx = trainval_idx[nidx:]

            # structure the indices as dicts {dataset_i : [piece_i,...,piece_i]}
            test_idx_dict = idx_tuple_to_dict(test_idx, self.datasets_map)
            train_idx_dict = idx_tuple_to_dict(train_idx, self.datasets_map)
            val_idx_dict = idx_tuple_to_dict(val_idx, self.datasets_map)

            # create the datasets
            self.dataset_train = ConcatDataset([self.datasets[k][train_idx_dict[k]] for k in train_idx_dict.keys()])
            self.dataset_val = ConcatDataset([self.datasets[k][val_idx_dict[k]] for k in val_idx_dict.keys()])
            self.dataset_test = ConcatDataset([self.datasets[k][test_idx_dict[k]] for k in test_idx_dict.keys()])
            print(f"Running evaluation on collections {self.test_collections}")
            print(
                f"Train size :{len(self.dataset_train)}, Val size :{len(self.dataset_val)}, Test size :{len(self.dataset_test)}"
            )
        # compute the ratio between real edges and potential edges
        # real_pot_ratios = list()
        # self.real_pot_ratio = sum([graph["truth_edges_mask"].shape[0]/torch.sum(graph["truth_edges_mask"]) for dataset in self.datasets for graph in dataset.graphs])/len(self.datasets_map)
        self.pot_real_ratio = sum([d.get_positive_weight() for d in self.datasets])/len(self.datasets)

    def train_dataloader(self):
        return PygDataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return PygDataLoader(
            self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return PygDataLoader(
            self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return self.test_dataloader()

    def num_dropped_truth_edges(self):
        return sum([d.num_dropped_truth_edges() for d in self.datasets])




def idx_tuple_to_dict(idx_tuple, datasets_map):
    """Transforms indices of a list of tuples of indices (dataset, piece_in_dataset) 
    into a dict {dataset: [piece_in_dataset,...,piece_in_dataset]}"""
    result_dict = defaultdict(list)
    for x in idx_tuple:
        result_dict[datasets_map[x][0]].append(datasets_map[x][1])
    return result_dict


class GraphMixVSDataModule(LightningDataModule):
    def __init__(
            self, batch_size=50, num_workers=4, force_reload=False, test_collections=None, pot_edges_max_dist=2, include_measures=False
    ):
        super(GraphMixVSDataModule, self).__init__()
        self.batch_size = batch_size
        self.bucket_boundaries = [200, 300, 400, 500, 700, 1000]
        self.num_workers = num_workers
        self.force_reload = force_reload
        self.include_measures = include_measures
        self.normalize_features = True
        self.datasets = [
            Bach370ChoralesGraphVoiceSeparationDataset(force_reload=self.force_reload, nprocs=self.num_workers, pot_edges_dist=pot_edges_max_dist, include_measures=self.include_measures),
            MCMAGraphVoiceSeparationDataset(force_reload=self.force_reload, nprocs=self.num_workers, pot_edges_dist=pot_edges_max_dist, include_measures=self.include_measures),
            HaydnStringQuartetGraphVoiceSeparationDataset(force_reload=self.force_reload, nprocs=self.num_workers, pot_edges_dist=pot_edges_max_dist, include_measures=self.include_measures),
            # MozartStringQuartetGraphVoiceSeparationDataset(force_reload=self.force_reload, nprocs=self.num_workers, pot_edges_dist=pot_edges_max_dist, include_measures=self.include_measures),
        ]
        if not (all([d.features == self.datasets[0].features for d in self.datasets])):
            raise Exception("Input dataset has different features, Datasets {} with sizes: {}".format(
                " ".join([d.name for d in self.datasets]), " ".join([str(d.features) for d in self.datasets])))
        self.features = self.datasets[0].features
        self.test_collections = test_collections

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.datasets_map = [(dataset_i, piece_i) for dataset_i, dataset in enumerate(self.datasets) for piece_i in
                             range(len(dataset))]
        if self.test_collections is None or self.test_collections == "split":
            idxs = range(len(self.datasets_map))
            collections = [self.datasets[self.datasets_map[i][0]].graphs[self.datasets_map[i][1]].collection for i
                           in idxs]
            trainval_idx, test_idx = train_test_split(idxs, test_size=0.3, stratify=collections, random_state=0)
            trainval_collections = [collections[i] for i in trainval_idx]
            train_idx, val_idx = train_test_split(trainval_idx, test_size=0.1, stratify=trainval_collections,
                                                  random_state=0)

            # structure the indices as dicts {dataset_i : [piece_i,...,piece_i]}
            self.test_idx_dict = idx_tuple_to_dict(test_idx, self.datasets_map)
            self.train_idx_dict = idx_tuple_to_dict(train_idx, self.datasets_map)
            self.val_idx_dict = idx_tuple_to_dict(val_idx, self.datasets_map)

            # create the datasets

            print("Running on all collections")
            print(
                f"Train size :{len(train_idx)}, Val size :{len(val_idx)}, Test size :{len(test_idx)}"
            )
        else:
            # idxs = torch.randperm(len(self.datasets_map)).long()
            idxs = range(len(self.datasets_map))
            test_idx = [
                i
                for i in idxs
                if self.datasets[self.datasets_map[i][0]].graphs[
                       self.datasets_map[i][1]].collection in self.test_collections
            ]
            trainval_idx = [i for i in idxs if i not in test_idx]
            trainval_collections = [
                self.datasets[self.datasets_map[i][0]].graphs[self.datasets_map[i][1]].collection for i in
                trainval_idx]
            train_idx, val_idx = train_test_split(trainval_idx, test_size=0.1, stratify=trainval_collections,
                                                  random_state=0)
            # nidx = int(len(trainval_idx) * 0.9)
            # train_idx = trainval_idx[:nidx]
            # val_idx = trainval_idx[nidx:]

            # structure the indices as dicts {dataset_i : [piece_i,...,piece_i]}
            self.test_idx_dict = idx_tuple_to_dict(test_idx, self.datasets_map)
            self.train_idx_dict = idx_tuple_to_dict(train_idx, self.datasets_map)
            self.val_idx_dict = idx_tuple_to_dict(val_idx, self.datasets_map)
            # create the datasets
            print(f"Running evaluation on collections {self.test_collections}")
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
        out["x"] = F.normalize(e["x"].squeeze(0).float()) if self.normalize_features else e["x"].squeeze(0).float()
        out["y"] = e["y"].squeeze(0)
        out["edge_index"] = e["edge_index"].squeeze(0)
        out["edge_type"] = e["edge_type"].squeeze(0)
        out["potential_edges"] = e["potential_edges"].squeeze(0)
        out["truth_edges"] = torch.tensor(e["truth_edges"].squeeze()).to(out["potential_edges"].device)
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
            x.append(e["x"])
            lengths.append(e["x"].shape[0])
            edge_index.append(e["edge_index"])
            edge_types.append(e["edge_type"])
            y.append(e["y"])
            note_array.append(torch.tensor(e["note_array"]))
            max_idx.append(e["x"].shape[0])
            potential_edges.append(e["potential_edges"])
            true_edges.append(torch.tensor(e["truth_edges"]).long())
        lengths = torch.tensor(lengths).long()
        out["lengths"], perm_idx = lengths.sort(descending=True)
        perm_idx = perm_idx.tolist()
        max_idx = np.cumsum(np.array([0] + [max_idx[i] for i in perm_idx]))
        out["x"] = torch.cat([x[i] for i in perm_idx], dim=0).float()
        out["edge_index"] = torch.cat([edge_index[pi]+max_idx[i] for i, pi in enumerate(perm_idx)], dim=1).long()
        out["potential_edges"] = torch.cat([potential_edges[pi]+max_idx[i] for i, pi in enumerate(perm_idx)], dim=1).long()
        out["true_edges"] = torch.cat([true_edges[pi]+max_idx[i] for i, pi in enumerate(perm_idx)], dim=1).long()
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
        for dataset in self.datasets:
            dataset.set_split("train")
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
        for dataset in self.datasets:
            dataset.set_split("validate")
        self.dataset_val = ConcatDataset([self.datasets[k][self.val_idx_dict[k]] for k in self.val_idx_dict.keys()])
        return torch.utils.data.DataLoader(
            self.dataset_val, batch_size=1, num_workers=self.num_workers, collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        for dataset in self.datasets:
            dataset.set_split("validate")
        self.dataset_test = ConcatDataset([self.datasets[k][self.test_idx_dict[k]] for k in self.test_idx_dict.keys()])
        return torch.utils.data.DataLoader(
            self.dataset_test, batch_size=1, num_workers=self.num_workers, collate_fn=self.collate_fn
        )


class AugmentedGraphDatamodule(LightningDataModule):
    def __init__(self, batch_size=1, num_workers=4, force_reload=False, include_synth=False, num_tasks=11, collection="all", version="v1.0.0", include_measures=False, transpose=False):
        super(AugmentedGraphDatamodule, self).__init__()
        self.bucket_boundaries = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.force_reload = force_reload
        self.normalize_features = True
        self.version = version
        self.include_measures = include_measures
        data_source = AugmentedNetChordGraphDataset(
            force_reload=self.force_reload, nprocs=self.num_workers,
            include_synth=include_synth, num_tasks=num_tasks, collection=collection, include_measures=include_measures, transpose=transpose
        ) if version=="v1.0.0" else Augmented2022ChordGraphDataset(
                    force_reload=self.force_reload, nprocs=self.num_workers,
                    include_synth=include_synth, num_tasks=num_tasks, collection=collection, include_measures=include_measures, transpose=transpose)
        self.datasets = [data_source]
        self.tasks = self.datasets[0].tasks
        if not (all([d.features == self.datasets[0].features for d in self.datasets])):
            raise Exception("Input dataset has different features, Datasets {} with sizes: {}".format(
                " ".join([d.name for d in self.datasets]), " ".join([str(d.features) for d in self.datasets])))
        self.features = self.datasets[0].features

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.datasets_map = [(dataset_i, piece_i) for dataset_i, dataset in enumerate(self.datasets) for piece_i in
                             range(len(dataset))]

        idxs = range(len(self.datasets_map))

        test_idx = [
            i
            for i in idxs
            if self.datasets[self.datasets_map[i][0]].graphs[
                   self.datasets_map[i][1]].collection == "test"
        ]

        # val_idx = [
        #     i
        #     for i in idxs
        #     if self.datasets[self.datasets_map[i][0]].graphs[
        #            self.datasets_map[i][1]].collection == "validation"
        # ]

        train_idx = [
            i
            for i in idxs
            if self.datasets[self.datasets_map[i][0]].graphs[
                   self.datasets_map[i][1]].collection == "training"
        ]

        # structure the indices as dicts {dataset_i : [piece_i,...,piece_i]}
        self.test_idx_dict = idx_tuple_to_dict(test_idx, self.datasets_map)
        self.train_idx_dict = idx_tuple_to_dict(train_idx, self.datasets_map)
        # self.val_idx_dict = idx_tuple_to_dict(val_idx, self.datasets_map)
        # create the datasets
        print(
            f"Train size :{len(train_idx)}, Val size :{len(test_idx)}, Test size :{len(test_idx)}"
        )

    def collate_fn(self, batch):
        e = batch[0]
        out = {}
        if self.include_measures:
            out["beat_nodes"] = e["beat_nodes"].long().squeeze()
            out["beat_edges"] = e["beat_edges"].long().squeeze()
            out["measure_nodes"] = e["measure_nodes"].long().squeeze()
            out["measure_edges"] = e["measure_edges"].long().squeeze()
        # batch_inputs = F.normalize(batch_inputs.squeeze(0)) if self.normalize_features else batch_inputs.squeeze(0)
        out["x"] = e["x"].squeeze(0).float()
        batch_labels = e["y"].squeeze(0)
        out["onset_div"] = e["onset_div"].squeeze().to(e["x"].device)
        out["note_array"] = e["note_array"].squeeze().to(e["x"].device)
        if self.version == "v1.0.0":
            from musgconv.utils.chord_representations import available_representations
        else:
            from musgconv.utils.chord_representations_latest import available_representations
        out["y"] = {task: batch_labels[:, i].squeeze().long() for i, task in enumerate(available_representations.keys())}
        out["y"]["onset"] = batch_labels[:, -1].squeeze()
        edges = e["edge_index"].squeeze(0)
        edge_type = e["edge_type"].squeeze(0)
        # Add reverse edges
        out["edge_index"], out["edge_type"] = add_reverse_edges_from_edge_index(edges, edge_type)
        # edges = torch.cat([edges, edges.flip(0)], dim=1)
        # edge_type = torch.cat([edge_type, edge_type], dim=0)
        return out

    def collate_train_fn(self, examples):
        out = {}
        lengths = list()
        x = list()
        edge_index = list()
        edge_types = list()
        y = list()
        onset_divs = list()
        max_idx = []
        max_onset_div = []
        beats = []
        beat_eindex = []
        measures = []
        measure_eindex = []
        note_array = []
        for e in examples:
            if self.include_measures:
                beats.append(e["beat_nodes"].long())
                beat_eindex.append(e["beat_edges"].long())
                measures.append(e["measure_nodes"].long())
                measure_eindex.append(e["measure_edges"].long())
            lengths.append(e["y"].shape[0])
            x.append(e["x"])
            edge_index.append(e["edge_index"])
            edge_types.append(e["edge_type"])
            y.append(e["y"])
            onset_divs.append(e["onset_div"])
            note_array.append(e["note_array"])
            max_idx.append(e["x"].shape[0])
            max_onset_div.append(e["onset_div"].max().item() + 1)
        lengths = torch.tensor(lengths).long()
        lengths, perm_idx = lengths.sort(descending=True)
        out["lengths"] = lengths
        perm_idx = perm_idx.tolist()
        max_idx = np.cumsum(np.array([0] + [max_idx[i] for i in perm_idx]))
        max_onset_div = np.cumsum(np.array([0] + [max_onset_div[i] for i in perm_idx]))
        out["x"] = torch.cat([x[i] for i in perm_idx], dim=0).float()
        out["edge_index"] = torch.cat([edge_index[pi]+max_idx[i] for i, pi in enumerate(perm_idx)], dim=1).long()
        out["edge_type"] = torch.cat([edge_types[i] for i in perm_idx], dim=0).long()
        # y = torch.cat([y[i] for i in perm_idx], dim=0).float()
        # batch_label = {task: y[:, i].squeeze().long() for i, task in
        #                enumerate(available_representations.keys())}
        # batch_label["onset"] = y[:, -1]
        y = torch.nn.utils.rnn.pad_sequence([y[i] for i in perm_idx], batch_first=True, padding_value=-1)
        if self.version == "v1.0.0":
            from musgconv.utils.chord_representations import available_representations
        else:
            from musgconv.utils.chord_representations_latest import available_representations
        out["y"] = {task: y[:, :, i].squeeze().long() for i, task in
                       enumerate(available_representations.keys())}
        out["y"]["onset"] = y[:, :, -1]
        out["onset_div"] = torch.cat([onset_divs[pi]+max_onset_div[i] for i, pi in enumerate(perm_idx)], dim=0).long()
        out["note_array"] = torch.cat([note_array[i] for i in perm_idx], dim=0).float()
        if self.include_measures:
            max_beat_idx = np.cumsum(np.array([0] + [beats[i].shape[0] for i in perm_idx]))
            out["beat_nodes"] = torch.cat([beats[pi] + max_beat_idx[i] for i, pi in enumerate(perm_idx)], dim=0).long()
            out["beat_edges"] = torch.cat(
                [torch.vstack((beat_eindex[pi][0] + max_idx[i], beat_eindex[pi][1] + max_beat_idx[i])) for i, pi in
                 enumerate(perm_idx)], dim=1).long()
            max_measure_idx = np.cumsum(np.array([0] + [measures[i].shape[0] for i in perm_idx]))
            out["measure_nodes"] = torch.cat([measures[pi] + max_measure_idx[i] for i, pi in enumerate(perm_idx)], dim=0).long()
            out["measure_edges"] = torch.cat(
                [torch.vstack((measure_eindex[pi][0] + max_idx[i], measure_eindex[pi][1] + max_measure_idx[i])) for
                 i, pi in enumerate(perm_idx)], dim=1).long()
        return out

    def train_dataloader(self):
        for dataset in self.datasets:
            dataset.set_split("train")
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
        self.dataset_test = ConcatDataset([self.datasets[k][self.test_idx_dict[k]] for k in self.test_idx_dict.keys()])
        return torch.utils.data.DataLoader(
            self.dataset_test, batch_size=1, num_workers=self.num_workers, collate_fn=self.collate_fn,
            drop_last=False, pin_memory=False,
        )

    def test_dataloader(self):
        self.dataset_test = ConcatDataset([self.datasets[k][self.test_idx_dict[k]] for k in self.test_idx_dict.keys()])
        return torch.utils.data.DataLoader(
            self.dataset_test, batch_size=1, num_workers=self.num_workers, collate_fn=self.collate_fn
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn
        )




