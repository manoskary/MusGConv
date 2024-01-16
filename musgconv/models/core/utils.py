import torch
from torch.nn import functional as F
import random
from torch_geometric.nn import SAGEConv, GATConv, ResGatedGraphConv, EdgeConv
import torch.nn as nn
from musgconv.models.core.hgnn import HeteroMusGConv


class SMOTE(object):
    """
    Minority Sampling with SMOTE.
    """
    def __init__(self, distance='custom', dims=512, k=2):
        super(SMOTE, self).__init__()
        self.newindex = 0
        self.k = k
        self.dims = dims
        self.distance_measure = distance

    def populate(self, N, i, nnarray, min_samples, k, device='cpu'):
        new_index = torch.arange(self.newindex, self.newindex + N, dtype=torch.int64, device=device)
        nn = torch.randint(0, k-1, (N, ), dtype=torch.int64, device=device)
        diff = min_samples[nnarray[nn]] - min_samples[i]
        gap = torch.rand((N, self.dims), dtype=torch.float32, device=device)
        self.synthetic_arr[new_index] = min_samples[i] + gap * diff
        self.newindex += N

    def k_neighbors(self, euclid_distance, k, device='cpu'):
        nearest_idx = torch.zeros((euclid_distance.shape[0], euclid_distance.shape[0]), dtype=torch.int64, device=device)

        idxs = torch.argsort(euclid_distance, dim=1)
        nearest_idx[:, :] = idxs

        return nearest_idx[:, 1:k+1]

    def find_k(self, X, k, device='cpu'):
        z = F.normalize(X, p=2, dim=1)
        distance = torch.mm(z, z.t())
        return self.k_neighbors(distance, k, device=device)

    # TODO: Need to find a matrix version of Euclid Distance and Cosine Distance.
    def find_k_euc(self, X, k, device='cpu'):
        euclid_distance = torch.zeros((X.shape[0], X.shape[0]), dtype=torch.float32, device=device)

        for i in range(len(X)):
            dif = (X - X[i]) ** 2
            dist = torch.sqrt(dif.sum(axis=1))
            euclid_distance[i] = dist

        return self.k_neighbors(euclid_distance, k, device=device)

    def find_k_cos(self, X, k, device='cpu'):
        cosine_distance = F.cosine_similarity(X, X)
        return self.k_neighbors(cosine_distance, k, device=device)

    def generate(self, min_samples, N, k, device='cpu'):
        """
        Returns (N/100) * n_minority_samples synthetic minority samples.
        Parameters
        ----------
        min_samples : Numpy_array-like, shape = [n_minority_samples, n_features]
            Holds the minority samples
        N : percetange of new synthetic samples:
            n_synthetic_samples = N/100 * n_minority_samples. Can be < 100.
        k : int. Number of nearest neighbours.
        Returns
        -------
        S : Synthetic samples. array,
            shape = [(N/100) * n_minority_samples, n_features].
        """
        T = min_samples.shape[0]
        self.synthetic_arr = torch.zeros(int(N / 100) * T, self.dims, dtype=torch.float32, device=device)
        N = int(N / 100)
        if self.distance_measure == 'euclidian':
            indices = self.find_k_euc(min_samples, k, device=device)
        elif self.distance_measure == 'cosine':
            indices = self.find_k_cos(min_samples, k, device=device)
        else:
            indices = self.find_k(min_samples, k, device=device)
        for i in range(indices.shape[0]):
            self.populate(N, i, indices[i], min_samples, k, device=device)
        self.newindex = 0
        return self.synthetic_arr

    def fit_generate(self, X, y):
        """
        Over-samples using SMOTE. Returns synthetic samples concatenated at the end of the original samples.
        Parameters
        ----------
        X : Numpy_array-like, shape = [n_samples, n_features]
            The input features
        y : Numpy_array-like, shape = [n_samples]
            The target labels.

        Returns
        -------
        X_resampled : Numpy_array, shape = [(n_samples + n_synthetic_samples), n_features]
            The array containing the original and synthetic samples.
        y_resampled : Numpy_array, shape = [(n_samples + n_synthetic_samples)]
            The corresponding labels of `X_resampled`.
        """
        # get occurence of each class
        occ = torch.eye(int(y.max() + 1), int(y.max() + 1), device=X.device)[y].sum(axis=0)
        # get the dominant class
        dominant_class = torch.argmax(occ)
        # get occurence of the dominant class
        n_occ = int(occ[dominant_class].item())
        for i in range(len(occ)):
            # For Mini-Batch Training exclude examples with less than k occurances in the mini banch.
            if i != dominant_class and occ[i] >= self.k:
                # calculate the amount of synthetic data to generate
                N = (n_occ - occ[i]) * 100 / occ[i]
                if N != 0:
                    candidates = X[y == i]
                    xs = self.generate(candidates, N, self.k, device=X.device)
                    X = torch.cat((X, xs))
                    ys = torch.ones(xs.shape[0], device=y.device) * i
                    y = torch.cat((y, ys))
        return X, y.long()



class SageEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers=2, dropout=0.5, activation=F.relu, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_channels, out_channels, aggr="sum"))
        if n_layers > 1:
            for i in range(n_layers-1):
                self.layers.append(SAGEConv(out_channels, out_channels, aggr="sum"))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def reset_parameters(self):
        for conv in self.layers:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_feature, **kwargs):
        for conv in self.layers[:-1]:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = F.normalize(x, dim=-1)
            x = self.dropout(x)
        x = self.layers[-1](x, edge_index)
        return x


class ResGatedConvEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers=2, dropout=0.5, activation=F.relu, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(ResGatedGraphConv(in_channels, out_channels))
        if n_layers > 1:
            for i in range(n_layers-1):
                self.layers.append(ResGatedGraphConv(out_channels, out_channels))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def reset_parameters(self):
        for conv in self.layers:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_feature, **kwargs):
        for conv in self.layers[:-1]:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = F.normalize(x, dim=-1)
            x = self.dropout(x)
        x = self.layers[-1](x, edge_index)
        return x


class GATEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers=2, dropout=0.5, activation=F.relu, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_channels, out_channels, aggr="sum"))
        if n_layers > 1:
            for i in range(n_layers - 1):
                self.layers.append(GATConv(out_channels, out_channels, aggr="sum"))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def reset_parameters(self):
        for conv in self.layers:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_feature, **kwargs):
        for conv in self.layers[:-1]:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = F.normalize(x, dim=-1)
            x = self.dropout(x)
        x = self.layers[-1](x, edge_index)
        return x


class EdgeConvEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers=2, dropout=0.5, activation=F.relu, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        mlp = nn.Sequential(
            nn.Linear(2*in_channels, out_channels),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Linear(out_channels, out_channels),
        )
        self.layers.append(EdgeConv(mlp, aggr="add"))
        if n_layers > 1:
            for i in range(n_layers - 1):
                mlp = nn.Sequential(
                    nn.Linear(2*out_channels, out_channels),
                    nn.ReLU(),
                    nn.BatchNorm1d(out_channels),
                    nn.Linear(out_channels, out_channels),
                )
                self.layers.append(EdgeConv(mlp,
                    aggr="add"))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def reset_parameters(self):
        for conv in self.layers:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_feature, **kwargs):
        for conv in self.layers[:-1]:
            x = conv(x, edge_index)
            x = self.activation(x)
            x = F.normalize(x, dim=-1)
            x = self.dropout(x)
        x = self.layers[-1](x, edge_index)
        return x


class HeteroMusGConvEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, metadata, n_layers=2, dropout=0.5, activation=F.relu, **kwargs):
        super().__init__()
        self.in_edge_features = kwargs.get("in_edge_features", 0)
        self.return_edge_emb = kwargs.get("return_edge_emb", False)
        aggregation = kwargs.get("aggregation", "cat")
        self.layers = nn.ModuleList()
        self.layers.append(HeteroMusGConv(in_channels, out_channels, metadata, in_edge_features=self.in_edge_features, return_edge_emb=self.return_edge_emb, aggregation=aggregation))
        if n_layers > 2:
            for i in range(n_layers - 2):
                self.layers.append(HeteroMusGConv(
                    out_channels, out_channels, metadata,
                    in_edge_features=(out_channels if self.return_edge_emb else 0),
                    return_edge_emb=self.return_edge_emb, aggregation=aggregation))
        self.layers.append(HeteroMusGConv(out_channels, out_channels, metadata, in_edge_features=(out_channels if self.return_edge_emb else 0), return_edge_emb=False, aggregation=aggregation))
        self.dropout = dropout
        self.activation = activation

    def reset_parameters(self):
        for conv in self.layers:
            conv.reset_parameters()

    def forward(self, x_dict, edge_index_dict, edge_feature_dict, **kwargs):
        for conv in self.layers[:-1]:
            if self.return_edge_emb:
                x_dict, edge_feature_dict = conv(x_dict, edge_index_dict, edge_feature_dict)
            else:
                x_dict = conv(x_dict, edge_index_dict, edge_feature_dict)
                edge_feature_dict = {k: None for k in edge_index_dict.keys()}
            x_dict = {k: self.activation(v) for k, v in x_dict.items()}
            x_dict = {k: F.normalize(v, dim=-1) for k, v in x_dict.items()}
            x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}
        x_dict = self.layers[-1](x_dict, edge_index_dict, edge_feature_dict)
        return x_dict