import numpy as np
from random import shuffle
from torch.utils.data import Sampler
import torch


class BySequenceLengthSampler(Sampler):

    def __init__(self, data_source,
                 bucket_boundaries, batch_size=64, drop_last=False):
        super(BySequenceLengthSampler, self).__init__(data_source)
        self.data_source = data_source
        ind_n_len = []
        for i, x in enumerate(data_source):
            ind_n_len.append((i, x["x"].shape[0]))

        self.ind_n_len = ind_n_len
        self.bucket_boundaries = bucket_boundaries
        self.batch_size = batch_size
        self.drop_last = drop_last

        if self.drop_last:
            print("WARNING: drop_last=True, dropping last non batch-size batch in every bucket ... ")

        self.boundaries = list(self.bucket_boundaries)
        self.buckets_min = torch.tensor([np.iinfo(np.int32).min] + self.boundaries)
        self.buckets_max = torch.tensor(self.boundaries + [np.iinfo(np.int32).max])
        self.boundaries = torch.tensor(self.boundaries)

    def shuffle_tensor(self, t):
        return t[torch.randperm(len(t))]

    def __iter__(self):
        data_buckets = dict()
        # where p is the id number and seq_len is the length of this id number.
        for p, seq_len in self.ind_n_len:
            pid = self.element_to_bucket_id(p, seq_len)
            if pid in data_buckets.keys():
                data_buckets[pid].append(p)
            else:
                data_buckets[pid] = [p]

        for k in data_buckets.keys():
            data_buckets[k] = torch.tensor(data_buckets[k])

        iter_list = []
        for k in data_buckets.keys():
            t = self.shuffle_tensor(data_buckets[k])
            batch = torch.split(t, self.batch_size, dim=0)
            if self.drop_last and len(batch[-1]) != self.batch_size:
                batch = batch[:-1]

            iter_list += batch

        shuffle(iter_list)  # shuffle all the batches so they arent ordered by bucket
        # size
        for i in iter_list:
            yield i.numpy().tolist()  # as it was stored in an array

    def __len__(self):
        return len(self.data_source)//self.batch_size

    def element_to_bucket_id(self, x, seq_length):
        valid_buckets = (seq_length >= self.buckets_min) * (seq_length < self.buckets_max)
        bucket_id = valid_buckets.nonzero()[0].item()
        return bucket_id


class SubgraphCreationSampler(Sampler):
    """
    This sampler is used to create subgraphs from a graph dataset.

    This sampler takes as input a graph dataset and creates subgraphs of a given size.
    It creates a different number of subgraphs based on the size of the original graph and the max subgraph size.
    It is used in the ASAPGraphDataset class.
    """
    def __init__(self, data_source, max_subgraph_size=100, drop_last=False, batch_size=64, train_idx=None, subgraphs_per_max_size:int=5):
        self.data_source = data_source
        bucket_boundaries = [2*max_subgraph_size, 5*max_subgraph_size, 10*max_subgraph_size, 20*max_subgraph_size]
        self.sampling_sizes = np.array([2, 4, 10, 20, 40])*subgraphs_per_max_size
        ind_n_len = []
        train_idx = train_idx if train_idx is not None else list()
        for i, g in enumerate(data_source.graphs):
            if i in train_idx:
                ind_n_len.append((i, g.x.shape[0]))
        self.ind_n_len = ind_n_len
        self.bucket_boundaries = bucket_boundaries
        self.drop_last = drop_last
        self.batch_size = batch_size
        if self.drop_last:
            print("WARNING: drop_last=True, dropping last non batch-size batch in every bucket ... ")
        self.max_size = max_subgraph_size
        self.boundaries = list(self.bucket_boundaries)
        self.buckets_min = torch.tensor([np.iinfo(np.int32).min] + self.boundaries)
        self.buckets_max = torch.tensor(self.boundaries + [np.iinfo(np.int32).max])
        self.boundaries = torch.tensor(self.boundaries)
        self.reindex_data()

    def reindex_data(self):
        data_buckets = dict()
        # where p is the id number and seq_len is the length of this id number.
        for p, seq_len in self.ind_n_len:
            pid = self.element_to_bucket_id(p, seq_len)
            if pid in data_buckets.keys():
                data_buckets[pid] += [p]*self.sampling_sizes[pid].item()
            else:
                data_buckets[pid] = [p]*self.sampling_sizes[pid].item()
        for k in data_buckets.keys():
            data_buckets[k] = torch.tensor(data_buckets[k])
        self.indices = torch.cat(list(data_buckets.values()))
        self.num_samples = len(self.indices)


    def __iter__(self):
        idx = self.indices[torch.multinomial(torch.ones(self.num_samples), self.num_samples, replacement=False)]
        batch = torch.split(idx, self.batch_size, dim=0)
        if self.drop_last and len(batch[-1]) != self.batch_size:
            batch = batch[:-1]
        for i in batch:
            yield i.numpy().tolist()

    def __len__(self):
        return self.num_samples // self.batch_size

    def element_to_bucket_id(self, x, seq_length):
        valid_buckets = (seq_length >= self.buckets_min) * (seq_length < self.buckets_max)
        bucket_id = valid_buckets.nonzero()[0].item()
        return bucket_id

