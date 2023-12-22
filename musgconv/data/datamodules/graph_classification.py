from torch_geometric.loader import DataLoader as PygDataLoader
from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import ConcatDataset
from musgconv.data.datasets import (
    ASAPGraphDataset,
    DCMLGraphDataset
)
from torch.nn import functional as F
from sklearn.model_selection import train_test_split
from musgconv.data.samplers import SubgraphCreationSampler
import numpy as np
from .mix_vs import idx_tuple_to_dict


class ComposerClassificationGraphDataModule(LightningDataModule):
    def __init__(
            self, batch_size=50, num_workers=4, force_reload=False, test_collections=None, max_size=200, include_measures=False, **kwargs
    ):
        super(ComposerClassificationGraphDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.force_reload = force_reload
        self.max_size = max_size
        # How many times should the subgraph be sampled for each graph compared to the integer division of the graph size by the subgraph size
        self.subgraph_ratio = kwargs.get("subgraph_ratio", 2)
        self.subgraph_size = max_size
        self.include_measures = include_measures
        self.normalize_features = True
        self.datasets = [
            # ASAPGraphDataset(force_reload=self.force_reload, n_jobs=self.num_workers, include_measures=self.include_measures, max_size=max_size),
            DCMLGraphDataset(force_reload=self.force_reload, n_jobs=self.num_workers, include_measures=self.include_measures, max_size=max_size)
        ]
        if not (all([d.features == self.datasets[0].features for d in self.datasets])):
            raise Exception("Input dataset has different features, Datasets {} with sizes: {}".format(
                " ".join([d.name for d in self.datasets]), " ".join([str(d.features) for d in self.datasets])))
        self.features = self.datasets[0].features
        self.n_classes = self.datasets[0].n_classes
        self.test_collections = test_collections if test_collections is not None else ["test"]

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.datasets_map = [(dataset_i, piece_i) for dataset_i, dataset in enumerate(self.datasets) for piece_i in
                             range(len(dataset))]

        # idxs = torch.randperm(len(self.datasets_map)).long()
        idxs = range(len(self.datasets_map))
        test_idx = [
            i
            for i in idxs
            if self.datasets[self.datasets_map[i][0]].graphs[
                   self.datasets_map[i][1]].y in self.test_collections
        ]
        if len(test_idx) == 0:
            traintest_idx = [i for i in idxs if i not in test_idx]
            traintest_idx_collections = [
                self.datasets[self.datasets_map[i][0]].graphs[self.datasets_map[i][1]].y for i in
                traintest_idx]
            trainval_idx, test_idx = train_test_split(traintest_idx, test_size=0.2, stratify=traintest_idx_collections,
                                                  random_state=0)
        trainval_idx = [i for i in idxs if i not in test_idx]
        trainval_collections = [
            self.datasets[self.datasets_map[i][0]].graphs[self.datasets_map[i][1]].y for i in
            trainval_idx]
        train_idx, val_idx = train_test_split(trainval_idx, test_size=0.1, stratify=trainval_collections,
                                              random_state=0)

        # structure the indices as dicts {dataset_i : [piece_i,...,piece_i]}
        self.test_idx_dict = idx_tuple_to_dict(test_idx, self.datasets_map)
        self.train_idx_dict = idx_tuple_to_dict(train_idx, self.datasets_map)
        self.val_idx_dict = idx_tuple_to_dict(val_idx, self.datasets_map)

        # create the datasets

        print(f"Train size :{len(train_idx)}, Val size :{len(val_idx)}, Test size :{len(test_idx)}")

    def collate_fn(self, batch):
        out = {}
        e = batch[0]
        out["x"] = F.normalize(e["x"].squeeze(0).float()) if self.normalize_features else e["x"].squeeze(0).float()
        out["y"] = torch.tensor([e["y"]]).long()
        out["edge_index"] = e["edge_index"].squeeze(0)
        out["edge_type"] = e["edge_type"].squeeze(0)
        out["note_array"] = torch.tensor(e["note_array"]).float()
        out["lengths"] = self.max_size
        return out

    def collate_train_fn(self, examples):
        out = {}
        lengths = list()
        x = list()
        edge_index = list()
        edge_types = list()
        y = list()
        note_array = list()
        max_idx = []
        beats = []
        beat_eindex = []
        measures = []
        measure_eindex = []
        for e in examples:

            x.append(e["x"])
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
        out["edge_index"] = torch.cat([edge_index[pi] + max_idx[i] for i, pi in enumerate(perm_idx)], dim=1).long()
        out["edge_type"] = torch.cat([edge_types[i] for i in perm_idx], dim=0).long()
        out["y"] = torch.tensor([y[i] for i in perm_idx]).long()
        out["note_array"] = torch.cat([note_array[i] for i in perm_idx], dim=0).float()
        return out

    def train_dataloader(self):
        print(f"Creating train dataloader with subgraph size {self.subgraph_size} and batch size {self.batch_size}")
        # compute training graphs here to change the graphs that exceed max size.
        training_graphs = [graph[0] for k in self.train_idx_dict.keys() for graph in
                           self.datasets[k][self.train_idx_dict[k]]]
        graph_sizes = np.array([g.num_nodes for g in training_graphs])
        # Compute the number of times each graph should be repeated based on size
        multiples = ((graph_sizes // self.subgraph_size) + 1)*self.subgraph_ratio
        # Create a list of indices repeating each graph the appropriate number of times
        indices = np.concatenate([np.repeat(i, m) for i, m in enumerate(multiples)])
        # Create the dataset by subgraphing each graph to the base size
        dataset_train = list()
        for idx in indices:
            g = training_graphs[idx]
            graph_length = g.num_nodes
            if graph_length > self.subgraph_size:  # subgraph only if the size is bigger than the subgraph size
                start = np.random.randint(0, graph_length - self.subgraph_size)
                sub_g = g.subgraph({"note": torch.arange(start, start + self.subgraph_size, dtype=torch.long)})
                dataset_train.append(sub_g)
            else:  # otherwise insert the entire graph
                dataset_train.append(g)
        print(f"Passing {len(dataset_train)} subgraphs into the dataloader")

        return PygDataLoader(dataset_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    # def train_dataloader(self):
    #     for dataset in self.datasets:
    #         dataset.set_split("train")
    #     sampler = SubgraphCreationSampler(self.datasets[0], max_subgraph_size=self.max_size,
    #                                       batch_size=self.batch_size, train_idx = self.train_idx_dict[0])
    #     # self.dataset_train = ConcatDataset([self.datasets[k][self.train_idx_dict[k]] for k in self.train_idx_dict.keys()])
    #     return torch.utils.data.DataLoader(
    #         self.datasets[0],
    #         batch_sampler=sampler,
    #         batch_size=1,
    #         num_workers=0,
    #         collate_fn=self.collate_train_fn,
    #         drop_last=False,
    #         pin_memory=False,
    #     )

    def val_dataloader(self):
        # for dataset in self.datasets:
        #     dataset.set_split("train")
        # sampler = SubgraphCreationSampler(self.datasets[0], max_subgraph_size=self.max_size,
        #                                   batch_size=self.batch_size, train_idx=self.val_idx_dict[0])
        # # self.dataset_train = ConcatDataset([self.datasets[k][self.train_idx_dict[k]] for k in self.train_idx_dict.keys()])
        # return torch.utils.data.DataLoader(
        #     self.datasets[0],
        #     batch_sampler=sampler,
        #     batch_size=1,
        #     num_workers=0,
        #     collate_fn=self.collate_train_fn,
        #     drop_last=False,
        #     pin_memory=False,
        # )

        # print(f"Creating val dataloader with subgraph size {self.subgraph_size} and batch size {self.batch_size}")
        # compute training graphs here to change the graphs that exceed max size.
        validation_graphs = [graph[0] for k in self.val_idx_dict.keys() for graph in
                           self.datasets[k][self.val_idx_dict[k]]]
        graph_sizes = np.array([g.num_nodes for g in validation_graphs])
        # Compute the number of times each graph should be repeated based on size
        multiples = ((graph_sizes // self.subgraph_size) + 1)*self.subgraph_ratio
        # Create a list of indices repeating each graph the appropriate number of times
        indices = np.concatenate([np.repeat(i, m) for i, m in enumerate(multiples)])
        # Create the dataset by subgraphing each graph to the base size
        dataset_train = list()
        for idx in indices:
            g = validation_graphs[idx]
            graph_length = g.num_nodes
            if graph_length > self.subgraph_size:  # subgraph only if the size is bigger than the subgraph size
                start = np.random.randint(0, graph_length - self.subgraph_size)
                sub_g = g.subgraph({"note": torch.arange(start, start + self.subgraph_size, dtype=torch.long)})
                dataset_train.append(sub_g)
            else:  # otherwise insert the entire graph
                dataset_train.append(g)
        print(f"Passing {len(dataset_train)} subgraphs into the dataloader")

        return PygDataLoader(dataset_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def test_dataloader(self):
        test_graphs = [graph[0] for k in self.test_idx_dict.keys() for graph in
                             self.datasets[k][self.test_idx_dict[k]]]
        graph_sizes = np.array([g.num_nodes for g in test_graphs])
        # Compute the number of times each graph should be repeated based on size
        multiples = ((graph_sizes // self.subgraph_size) + 1)*self.subgraph_ratio
        # Create a list of indices repeating each graph the appropriate number of times
        indices = np.concatenate([np.repeat(i, m) for i, m in enumerate(multiples)])
        # Create the dataset by subgraphing each graph to the base size
        dataset_train = list()
        for idx in indices:
            g = test_graphs[idx]
            graph_length = g.num_nodes
            if graph_length > self.subgraph_size:  # subgraph only if the size is bigger than the subgraph size
                start = np.random.randint(0, graph_length - self.subgraph_size)
                sub_g = g.subgraph({"note": torch.arange(start, start + self.subgraph_size, dtype=torch.long)})
                dataset_train.append(sub_g)
            else:  # otherwise insert the entire graph
                dataset_train.append(g)

        return PygDataLoader(dataset_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)