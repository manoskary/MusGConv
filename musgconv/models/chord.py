import torch
import copy
from musgconv.models.core import *
from torch_scatter import scatter_add
from pytorch_lightning import LightningModule
from musgconv.metrics.eval import MultitaskAccuracy
import pandas as pd
from musgconv.models.core.hgnn import MetricalGNN
from musgconv.utils import add_reverse_edges_from_edge_index
from musgconv.utils import METADATA
from torch_geometric.nn import to_hetero
from musgconv.models.core.utils import HeteroMusGConvEncoder, SageEncoder, GATEncoder, ResGatedConvEncoder


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss function

    Learning weights for each task according to the paper:
        Liebel L, Körner M. Auxiliary tasks in multi-task learning[J]
    """
    def __init__(self, tasks: list, loss_ft: nn.ModuleDict, loss_weights: dict = None, requires_grad=True):
        super(MultiTaskLoss, self).__init__()
        assert (set(tasks) == set(loss_ft.keys()))
        if loss_weights is not None:
            assert (set(tasks) == set(loss_weights.keys()))
            self.loss_weights = loss_weights
        else:
            self.loss_weights = {task: 1 for task in tasks}
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.requires_grad = requires_grad
        if requires_grad:
            self.params = nn.Parameter(torch.ones(len(tasks), requires_grad=True))
        else:
            self.params = torch.ones(len(tasks), requires_grad=False)

    def forward(self, pred, gt):
        out = {task: self.loss_ft[task](pred[task], gt[task]) for task in self.tasks}
        loss_sum = 0
        for i, loss in enumerate(out.values()):
            if self.requires_grad:
                loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
            else:
                loss_sum += loss
        out["total"] = loss_sum
        # out['total'] = torch.sum(torch.stack([self.loss_weights[t] * out[t] for t in self.tasks]))
        return out


class SimpleOnsetEdgePooling(nn.Module):
    def __init__(self, in_channels, dropout=0, add_to_edge_score=0.5):
        super(SimpleOnsetEdgePooling, self).__init__()
        self.in_channels = in_channels
        self.add_to_edge_score = add_to_edge_score
        self.dropout = dropout
        self.lin = nn.Linear(2 * in_channels, 1)
        self.trans = nn.Linear(in_channels, in_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.trans.reset_parameters()

    @staticmethod
    def compute_edge_score(raw_edge_score, edge_index, num_nodes):
        return torch.sigmoid(raw_edge_score)

    def forward(self, x, edge_index, batch):
        r"""Forward computation which computes the raw edge score, normalizes
        it, and merges the edges.
        Args:
            x (Tensor): The node features.
            edge_index (LongTensor): The edge indices.
            batch (LongTensor): Batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each node to a specific example.
        Return types:
            * **x** *(Tensor)* - The pooled node features.
        """
        # Run nodes on each edge through a linear layer
        e = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        e = self.lin(e).view(-1)
        e = F.dropout(e, p=self.dropout, training=self.training)
        e = self.compute_edge_score(e, edge_index, x.size(0))
        e = e + self.add_to_edge_score
        adj = torch.sparse_coo_tensor(
            edge_index, torch.ones(len(edge_index[0])).to(x.get_device()), (len(x), len(x))).to_dense().to(
            x.get_device())
        # Get neighbor information
        # h = x + torch.mm(adj, self.trans(x)) / (adj.sum(dim=1).reshape(adj.shape[0], -1) + 1)
        x, edge_index, batch = self.__merge_edges__(x, edge_index, batch, e)
        return x, edge_index.long(), batch

    def __merge_edges__(self, x, edge_index, batch, edge_score):
        nodes_remaining = set(range(x.size(0)))
        nodes_discarded = nodes_remaining(set(torch.unique(edge_index[0]).tolist()), nodes_remaining)
        cluster = torch.empty_like(batch, device=torch.device('cpu'))
        edge_argsort = edge_score.detach().cpu().numpy().argsort(kind='stable')[::-1]  # Use stable sort

        # Iterate through all edges, selecting it if it is not incident to
        # another already chosen edge.
        i = 0
        new_edge_indices = []
        edge_index_cpu = edge_index.cpu()
        for edge_idx in edge_argsort.tolist():
            source = edge_index_cpu[0, edge_idx].item()
            # I source not in the remaining nodes move to the next edge.
            if source not in nodes_remaining:
                continue

            target = edge_index_cpu[1, edge_idx].item()
            # If target is not in the remaining nodes move to the next edge.
            if target not in nodes_remaining:
                continue

            # if target is not in the discarded nodes move to the next edge.
            if target not in nodes_discarded:
                continue

            cluster[source] = i
            nodes_remaining.remove(source)
            nodes_discarded.remove(target)

            if source != target:
                cluster[target] = i
                nodes_remaining.remove(target)

            i += 1

        # The remaining nodes are simply kept.
        for node_idx in nodes_remaining:
            cluster[node_idx] = i
            i += 1
        cluster = cluster.to(x.device)

        # We compute the new features as an addition of the old ones.
        new_x = scatter_add(x, cluster, dim=0, dim_size=i)
        new_edge_score = edge_score[new_edge_indices]
        if len(nodes_remaining) > 0:
            remaining_score = x.new_ones(
                (new_x.size(0) - len(new_edge_indices),))
            new_edge_score = torch.cat([new_edge_score, remaining_score])
        new_x = new_x * new_edge_score.view(-1, 1)
        return new_x


class OnsetEdgePoolingVersion2(nn.Module):
    def __init__(self, in_channels, dropout=0):
        super(OnsetEdgePoolingVersion2, self).__init__()
        self.in_channels = in_channels
        self.trans = nn.Linear(in_channels, in_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.trans.reset_parameters()

    def forward(self, x, edge_index, idx=None):
        """Forward computation which computes the raw edge score, normalizes
        it, and merges the edges.
        Args:
            x (Tensor): The node features.
            edge_index (LongTensor): The edge indices.
        Return types:
            * **x** *(Tensor)* - The pooled node features.
        """
        device = x.get_device() if x.get_device()>=0 else "cpu"
        # if device >= 0:
        #     adj = torch.sparse_coo_tensor(
        #         edge_index, torch.ones(len(edge_index[0])).to(device), (len(x), len(x))).to_dense().to(device)
        # else:
        #     adj = torch.sparse_coo_tensor(
        #         edge_index, torch.ones(len(edge_index[0])), (len(x), len(x))).to_dense().type(x.dtype)
        # adj = adj.fill_diagonal_(1)
        # h = torch.mm(adj, self.trans(x)) / adj.sum(dim=1).reshape(adj.shape[0], -1)
        # add self loops to edge_index with size (2, num_edges + num_nodes)
        edge_index_sl = torch.cat([edge_index, torch.arange(x.size(0)).view(1, -1).repeat(2, 1).to(device)], dim=1)
        h = scatter(self.trans(x)[edge_index_sl[0]], edge_index_sl[1], 0, out=torch.zeros(x.shape).to(device), reduce='mean')
        if idx is not None:
            out = h[idx]
        else:
            out, idx = self.__merge_edges__(h, edge_index)
        return out, idx

    def __merge_edges__(self, x, edge_index):
        nodes_remaining = torch.ones(x.size(0), dtype=torch.long)
        nodes_discarded = torch.zeros(x.size(0), dtype=torch.long)
        nodes_discarded[torch.unique(edge_index[0])] = 1

        # Iterate through all edges, selecting it if it is not incident to
        # another already chosen edge.
        for edge_idx in range(edge_index.shape[-1]):
            source = edge_index[0, edge_idx].item()
            # I source not in the remaining nodes move to the next edge.
            if not nodes_remaining[source].item():
                continue

            target = edge_index[1, edge_idx].item()
            # If target is not in the remaining nodes move to the next edge.
            if not nodes_remaining[target].item():
                continue

            # if target is not in the discarded nodes move to the next edge.
            if not nodes_discarded[target].item():
                continue

            if source == target:
                continue


            # remove the source node from the remaining nodes
            nodes_remaining[source] = 0
            # remove the target node from the discarded nodes
            nodes_discarded[target] = 0

        # We compute the new features by trimming with the remaining nodes.
        new_x = x[nodes_remaining == 1]
        return new_x, nodes_remaining


class NadeClf(nn.Module):
    def __init__(self, input_size, n_hidden, out_dim, n_layers, activation=F.relu, dropout=0.5):
        super(NadeClf, self).__init__()
        self.VtoH = nn.Linear(out_dim, input_size, bias=False)
        self.HtoV = MLP(input_size, n_hidden, out_dim, n_layers=n_layers, bias=True, activation=activation, dropout=dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.VtoH.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, h):
        out = self.HtoV(h)
        h = h + self.VtoH(torch.sigmoid(out))
        return out, h


class NadeClassifierLayer(nn.Module):
    def __init__(self, input_size, n_hidden, tasks, n_layers, activation=F.relu, dropout=0.5):
        super(NadeClassifierLayer, self).__init__()
        self.tasks = tasks
        self.classifier = nn.ModuleDict({task: NadeClf(input_size, n_hidden, tdim, n_layers, activation, dropout) for task, tdim in tasks.items()})

    def forward(self, h):
        prediction = {}
        for task in self.tasks.keys():
            prediction[task], h = self.classifier[task](h)
        return prediction


class MultiTaskMLP(nn.Module):
    def __init__(self, in_feats, n_hidden, tasks: dict, n_layers, activation=F.relu, dropout=0.5):
        super(MultiTaskMLP, self).__init__()
        self.dropout = dropout
        self.n_layers = n_layers
        self.tasks = tasks
        self.classifier = nn.ModuleDict(
            {task: MLP(in_feats, n_hidden, tdim, n_layers, activation, dropout) for task, tdim in tasks.items()}
        )

    def reset_parameters(self):
        for task in self.tasks.keys():
            self.classifier[task].reset_parameters()

    def forward(self, x):
        prediction = {}
        for task in self.tasks.keys():
            prediction[task] = self.classifier[task](x)
        return prediction


class ChordEncoder(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, activation=F.relu, dropout=0.5, use_jk=False):
        super(ChordEncoder, self).__init__()
        self.activation = activation
        self.spelling_embedding = nn.Embedding(49, 16)
        self.pitch_embedding = nn.Embedding(128, 16)
        self.embedding = nn.Linear(in_feats-3, 32)
        # self.embedding = nn.Linear(in_feats-1, n_hidden)
        self.encoder = HGCN(64, n_hidden*2, n_hidden, n_layers, activation=activation, dropout=dropout, jk=use_jk)
        # self.encoder = HResGatedConv(64, n_hidden*2, n_hidden, n_layers, activation=activation, dropout=dropout, jk=use_jk)
        # self.etypes = {"onset":0, "consecutive":1, "during":2, "rest":3, "consecutive_rev":4, "during_rev":5, "rest_rev":6}
        # self.encoder = HeteroResGatedGraphConvLayer(n_hidden, n_hidden, etypes=self.etypes, reduction="none")
        # self.reduction = HeteroAttention(n_hidden, len(self.etypes.keys()))
        self.pool = OnsetEdgePoolingVersion2(n_hidden, dropout=dropout)
        self.proj1 = nn.Linear(n_hidden+1, n_hidden)
        self.layernorm1 = nn.BatchNorm1d(n_hidden)
        self.proj2 = nn.Linear(n_hidden, n_hidden//2)
        self.layernorm2 = nn.BatchNorm1d(n_hidden//2)
        self.gru = nn.GRU(input_size=n_hidden//2, hidden_size=int(n_hidden/2), num_layers=2, bidirectional=True,
                          batch_first=True, dropout=dropout)
        self.layernormgru = nn.LayerNorm(n_hidden)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.proj1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.proj2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.embedding.bias, 0)
        nn.init.constant_(self.proj1.bias, 0)
        nn.init.constant_(self.proj2.bias, 0)
        self.layernormgru.reset_parameters()
        self.layernorm1.reset_parameters()
        self.layernorm2.reset_parameters()
        self.encoder.reset_parameters()
        self.pool.reset_parameters()

    def forward(self, batch):
        x, edge_index, edge_type, onset_index, onset_idx, lengths = batch
        h_pitch = self.pitch_embedding(x[:, 0].long())
        h_spelling = self.spelling_embedding(x[:, 1].long())
        h = self.embedding(x[:, 2:-1])
        h = torch.cat([h, h_pitch, h_spelling], dim=-1)
        # h = F.normalize(self.embedding(x[:, :-1]))
        h = self.encoder(h, edge_index, edge_type)
        h = F.normalize(self.activation(h))
        h, idx = self.pool(h, onset_index, onset_idx)
        h = torch.cat([h, x[:, -1][idx].unsqueeze(-1)], dim=-1)
        h = self.layernorm1(self.activation(self.proj1(h)))
        h = self.layernorm2(self.activation(self.proj2(h)))
        if lengths is not None:
            lengths = lengths.tolist()
            h = torch.split(h, lengths, dim=0)
            h = nn.utils.rnn.pad_sequence(h, batch_first=True, padding_value=-1)
            h = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True)
            h, _ = self.gru(h)
            h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, padding_value=-1)
            h = self.layernormgru(h)
            h = h.reshape(-1, h.shape[-1])
        else:
            h, _ = self.gru(h.unsqueeze(0))
            h = self.layernormgru(h.squeeze(0))
        return h


class ChordEncoder(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, activation=F.relu, dropout=0.5, use_jk=False):
        super(ChordEncoder, self).__init__()
        self.activation = activation
        self.spelling_embedding = nn.Embedding(49, 16)
        self.pitch_embedding = nn.Embedding(128, 16)
        self.embedding = nn.Linear(in_feats-3, 32)
        # self.embedding = nn.Linear(in_feats-1, n_hidden)
        self.encoder = HGCN(64, n_hidden*2, n_hidden, n_layers, activation=activation, dropout=dropout, jk=use_jk)
        # self.encoder = HResGatedConv(64, n_hidden*2, n_hidden, n_layers, activation=activation, dropout=dropout, jk=use_jk)
        # self.etypes = {"onset":0, "consecutive":1, "during":2, "rest":3, "consecutive_rev":4, "during_rev":5, "rest_rev":6}
        # self.encoder = HeteroResGatedGraphConvLayer(n_hidden, n_hidden, etypes=self.etypes, reduction="none")
        # self.reduction = HeteroAttention(n_hidden, len(self.etypes.keys()))
        self.pool = OnsetEdgePoolingVersion2(n_hidden, dropout=dropout)
        self.proj1 = nn.Linear(n_hidden+1, n_hidden)
        self.layernorm1 = nn.BatchNorm1d(n_hidden)
        self.proj2 = nn.Linear(n_hidden, n_hidden//2)
        self.layernorm2 = nn.BatchNorm1d(n_hidden//2)
        self.gru = nn.GRU(input_size=n_hidden//2, hidden_size=int(n_hidden/2), num_layers=2, bidirectional=True,
                          batch_first=True, dropout=dropout)
        self.layernormgru = nn.LayerNorm(n_hidden)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.proj1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.proj2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.embedding.bias, 0)
        nn.init.constant_(self.proj1.bias, 0)
        nn.init.constant_(self.proj2.bias, 0)
        self.layernormgru.reset_parameters()
        self.layernorm1.reset_parameters()
        self.layernorm2.reset_parameters()
        self.encoder.reset_parameters()
        self.pool.reset_parameters()

    def forward(self, batch):
        x, edge_index, edge_type, onset_index, onset_idx, lengths = batch
        h_pitch = self.pitch_embedding(x[:, 0].long())
        h_spelling = self.spelling_embedding(x[:, 1].long())
        h = self.embedding(x[:, 2:-1])
        h = torch.cat([h, h_pitch, h_spelling], dim=-1)
        # h = F.normalize(self.embedding(x[:, :-1]))
        h = self.encoder(h, edge_index, edge_type)
        h = F.normalize(self.activation(h))
        h, idx = self.pool(h, onset_index, onset_idx)
        h = torch.cat([h, x[:, -1][idx].unsqueeze(-1)], dim=-1)
        h = self.layernorm1(self.activation(self.proj1(h)))
        h = self.layernorm2(self.activation(self.proj2(h)))
        if lengths is not None:
            lengths = lengths.tolist()
            h = torch.split(h, lengths, dim=0)
            h = nn.utils.rnn.pad_sequence(h, batch_first=True, padding_value=-1)
            h = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True)
            h, _ = self.gru(h)
            h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, padding_value=-1)
            h = self.layernormgru(h)
            h = h.reshape(-1, h.shape[-1])
        else:
            h, _ = self.gru(h.unsqueeze(0))
            h = self.layernormgru(h.squeeze(0))
        return h


class MetricalChordEncoder(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, activation=F.relu, dropout=0.5, use_jk=False, metrical=False, use_reledge=False, **kwargs):
        super(MetricalChordEncoder, self).__init__()
        self.activation = activation
        self.spelling_embedding = nn.Embedding(49, 16)
        self.pitch_embedding = nn.Embedding(128, 16)
        self.embedding = nn.Linear(in_feats-3, 32)
        self.etypes = {"onset": 0, "consecutive": 1, "during": 2, "rest": 3, "consecutive_rev": 4, "during_rev": 5,
                       "rest_rev": 6}
        # self.embedding = nn.Linear(in_feats-1, n_hidden)
        pitch_embeddding = kwargs.get("pitch_embedding", 0)
        pitch_embeddding = 0 if pitch_embeddding is None else pitch_embeddding
        return_edge_emb = kwargs.get("return_edge_emb", False)
        block = kwargs.get("conv_block", "SageConv")
        if block == "ResConv":
            print("Using ResGatedGraphConv")
            enc = ResGatedConvEncoder(64+pitch_embeddding, n_hidden, n_layers=n_layers, dropout=dropout, activation=activation)
            self.encoder = to_hetero(enc, metadata=METADATA, aggr="mean")
        elif block == "SageConv" or block == "Sage" or block is None:
            print("Using SageConv")
            enc = SageEncoder(64+pitch_embeddding, n_hidden, n_layers=n_layers, dropout=dropout, activation=activation)
            self.encoder = to_hetero(enc, metadata=METADATA, aggr="mean")
        elif block == "GAT" or block == "GATConv":
            print("Using GATConv")
            enc = GATEncoder(64+pitch_embeddding, n_hidden, n_layers=n_layers, dropout=dropout, activation=activation)
            self.encoder = to_hetero(enc, metadata=METADATA, aggr="mean")
        elif block == "RelEdgeConv":
            self.encoder = HeteroMusGConvEncoder(64+pitch_embeddding, n_hidden, metadata=METADATA, n_layers=n_layers, dropout=dropout, activation=activation,
                                               in_edge_features=pitch_embeddding, return_edge_emb=return_edge_emb)
        else:
            raise ValueError("Block type not supported")
        kwargs["conv_block"] = block

        # self.encoder = HResGatedConv(64, n_hidden*2, n_hidden, n_layers, activation=activation, dropout=dropout, jk=use_jk)
        #
        # self.encoder = HeteroResGatedGraphConvLayer(n_hidden, n_hidden, etypes=self.etypes, reduction="none")
        # self.reduction = HeteroAttention(n_hidden, len(self.etypes.keys()))
        self.pool = OnsetEdgePoolingVersion2(n_hidden, dropout=dropout)
        self.proj1 = nn.Linear(n_hidden+1, n_hidden)
        self.layernorm1 = nn.BatchNorm1d(n_hidden)
        self.proj2 = nn.Linear(n_hidden, n_hidden//2)
        self.layernorm2 = nn.BatchNorm1d(n_hidden//2)
        self.gru = nn.GRU(input_size=n_hidden//2, hidden_size=int(n_hidden/2), num_layers=2, bidirectional=True,
                          batch_first=True, dropout=dropout)
        self.layernormgru = nn.LayerNorm(n_hidden)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embedding.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.proj1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.proj2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.embedding.bias, 0)
        nn.init.constant_(self.proj1.bias, 0)
        nn.init.constant_(self.proj2.bias, 0)
        self.layernormgru.reset_parameters()
        self.layernorm1.reset_parameters()
        self.layernorm2.reset_parameters()
        self.encoder.reset_parameters()
        self.pool.reset_parameters()

    def forward(self, x_dict, edge_index_dict, edge_feature_dict, onset_index, onset_idx, lengths, **kwargs):
        x = x_dict["note"]
        h_pitch = self.pitch_embedding(x[:, 0].long())
        h_spelling = self.spelling_embedding(x[:, 1].long())
        h = self.embedding(x[:, 2:-1])
        h = torch.cat([h, h_pitch, h_spelling], dim=-1)
        x_dict = {"note": h}
        h = self.encoder(x_dict, edge_index_dict, edge_feature_dict, **kwargs)["note"]
        h = F.normalize(self.activation(h))
        h, idx = self.pool(h, onset_index, onset_idx)
        h = torch.cat([h, x[:, -1][idx].unsqueeze(-1)], dim=-1)
        h = self.layernorm1(self.activation(self.proj1(h)))
        h = self.layernorm2(self.activation(self.proj2(h)))
        if lengths is not None:
            lengths = lengths.tolist()
            h = torch.split(h, lengths, dim=0)
            h = nn.utils.rnn.pad_sequence(h, batch_first=True, padding_value=-1)
            h = nn.utils.rnn.pack_padded_sequence(h, lengths, batch_first=True)
            h, _ = self.gru(h)
            h, _ = nn.utils.rnn.pad_packed_sequence(h, batch_first=True, padding_value=-1)
            h = self.layernormgru(h)
            h = h.reshape(-1, h.shape[-1])
        else:
            h, _ = self.gru(h.unsqueeze(0))
            h = self.layernormgru(h.squeeze(0))
        return h



class MetricalChordPredictionModel(nn.Module):
    def __init__(self, in_feats, n_hidden=256, tasks: dict = {
        "localkey": 38, "tonkey": 38, "degree1": 22, "degree2": 22, "quality": 11, "inversion": 4,
        "root": 35, "romanNumeral": 31, "hrhythm": 7, "pcset": 121, "bass": 35, "tenor": 35,
        "alto": 35, "soprano": 35}, n_layers=1, activation=F.relu, dropout=0.5, use_nade=False, use_jk=False,
                 metrical=False, use_reledge=False, **kwargs):
        super(MetricalChordPredictionModel, self).__init__()
        self.dropout = dropout
        self.n_layers = n_layers
        self.tasks = tasks
        self.encoder = MetricalChordEncoder(in_feats, n_hidden, n_layers, activation=activation, dropout=dropout,
                                            use_jk=use_jk, metrical=metrical, use_reledge=use_reledge, **kwargs)
        if use_nade:
            self.classifier = NadeClassifierLayer(n_hidden, n_hidden, tasks=tasks, n_layers=1, activation=activation, dropout=dropout)
        else:
            self.classifier = MultiTaskMLP(n_hidden, n_hidden, tasks=tasks, n_layers=1, activation=activation, dropout=dropout)

    def forward(self, x_dict, edge_index_dict, edge_feature_dict, onset_edges, onset_idx, lengths, **kwargs):
        """
        Forward pass of the model.

        Parameters
        ----------
        x_dict: torch.Tensor
            (n_nodes, n_feats)
        edge_index_dict: Long Tensor
            (2, n_edges)
        edge_feature_dict: Long Tensor
            (n_edges, )
        onset_edges: Long Tensor
            (2, n_onset_edges)
        onset_idx: Long Tensor
            (n_onsets, )
        lengths: Long Tensor
            (n_sequences, )
        beat_nodes: Long Tensor
            (n_beats, )
        beat_edges: Long Tensor
            (2, n_beat_edges)
        measure_nodes: Long Tensor
            (n_measures, )
        measure_edges: Long Tensor
            (2, n_measure_edges)

        """
        h = self.encoder(x_dict, edge_index_dict, edge_feature_dict, onset_edges, onset_idx, lengths=lengths, **kwargs)
        prediction = self.classifier(h)
        return prediction


class ChordPredictionModel(nn.Module):
    def __init__(self, in_feats, n_hidden=256, tasks: dict = {
        "localkey": 38, "tonkey": 38, "degree1": 22, "degree2": 22, "quality": 11, "inversion": 4,
        "root": 35, "romanNumeral": 31, "hrhythm": 7, "pcset": 121, "bass": 35, "tenor": 35,
        "alto": 35, "soprano": 35}, n_layers=1, activation=F.relu, dropout=0.5, use_nade=False, use_jk=False):
        super(ChordPredictionModel, self).__init__()
        self.dropout = dropout
        self.n_layers = n_layers
        self.tasks = tasks
        self.encoder = ChordEncoder(in_feats, n_hidden, n_layers, activation=activation, dropout=dropout, use_jk=use_jk)
        if use_nade:
            self.classifier = NadeClassifierLayer(n_hidden, n_hidden, tasks=tasks, n_layers=1, activation=activation, dropout=dropout)
        else:
            self.classifier = MultiTaskMLP(n_hidden, n_hidden, tasks=tasks, n_layers=1, activation=activation, dropout=dropout)

    def forward(self, batch):
        x, edge_index, edge_type, onset_index, onset_idx, lengths = batch
        h = self.encoder((x, edge_index, edge_type, onset_index, onset_idx, lengths))
        prediction = self.classifier(h)
        return prediction

    def predict(self, score):
        from musgconv.utils import hetero_graph_from_note_array, select_features, add_reverse_edges_from_edge_index
        note_array = score.note_array(include_time_signature=True, include_pitch_spelling=True)
        onsets = torch.unique(torch.tensor(note_array["onset_beat"]))
        unique_onset_divs = torch.unique(torch.tensor(note_array["onset_div"]))
        measures = torch.tensor([[m.start.t, m.end.t] for m in score.parts[0].measures])
        measure_names = [m.number for m in score.parts[0].measures]
        s_measure = torch.zeros((len(unique_onset_divs)))
        for idx, measure_num in enumerate(measure_names):
            s_measure[torch.where((unique_onset_divs >= measures[idx, 0]) & (unique_onset_divs < measures[idx, 1]))] = measure_num
        nodes, edges = hetero_graph_from_note_array(note_array=note_array)
        note_features = select_features(note_array, "chord")
        onset_idx = unique_onsets(torch.tensor(note_array["onset_div"]))
        edge_index = torch.tensor(edges[:2, :]).long()
        x = torch.tensor(note_features).float()
        edge_type = torch.tensor(edges[2, :]).long()
        onset_edges = edge_index[:, edge_type == 0]
        edge_index, edge_type = add_reverse_edges_from_edge_index(edge_index, edge_type)
        onset_predictions = self.forward((x, edge_index, edge_type, onset_edges, onset_idx, None))
        onset_predictions["onset"] = onsets
        onset_predictions["s_measure"] = s_measure
        return onset_predictions


class PostProcessingMLTModel(nn.Module):
    """
    Post-processing module for the MLT model using a sequential model.
    """
    def __init__(self, tasks, n_hidden, n_layers, activation=F.relu, dropout=0.0):
        super().__init__()
        self.tasks = tasks
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        logits_dim = sum(list(self.tasks.values()))
        self.lstm = nn.LSTM(logits_dim, n_hidden, n_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(n_hidden*2, n_hidden)
        self.clf = nn.ModuleDict({task: nn.Linear(n_hidden, n_classes) for task, n_classes in self.tasks.items()})
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, x):
        h = torch.cat([F.softmax(x[task], dim=-1) for task in self.tasks.keys()], dim=-1)
        if len(h.shape) == 2:
            h = h.unsqueeze(0)
        h, _ = self.lstm(h)
        h = self.activation(self.fc(h.squeeze()))
        h = F.normalize(h, p=2, dim=-1)
        h = self.dropout(h)
        h = {task: self.clf[task](h) for task in self.tasks.keys()}
        return h

    def predict(self, x):
        h = {k: v for k, v in x.items() if k in self.tasks.keys()}
        onset_predictions = self.forward(h)
        onset_predictions["onset"] = x["onset"]
        onset_predictions["s_measure"] = x["s_measure"]
        return onset_predictions


class ChordPrediction(LightningModule):
    def __init__(self,
                 in_feats: int,
                 n_hidden: int,
                 tasks: dict,
                 n_layers: int,
                 activation=F.relu,
                 dropout=0.5,
                 lr=0.0001,
                 weight_decay=5e-4,
                 use_nade=False,
                 use_jk=False,
                 use_rotograd=False,
                 use_gradnorm=False,
                 weight_loss=True,
                 device=0
                 ):
        super(ChordPrediction, self).__init__()
        self.tasks = tasks
        self.save_hyperparameters()
        self.num_tasks = len(tasks.keys())
        self.module = ChordPredictionModel(
            in_feats, n_hidden, tasks, n_layers, activation, dropout,
            use_nade=use_nade, use_jk=use_jk).float().to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.test_roman = list()
        self.test_roman_ts = list()
        self.train_loss = MultiTaskLoss(
            list(tasks.keys()), nn.ModuleDict({task: nn.CrossEntropyLoss(ignore_index=-1) for task in tasks.keys()}), requires_grad=weight_loss)
        self.val_loss = MultiTaskLoss(
            list(tasks.keys()), nn.ModuleDict({task: nn.CrossEntropyLoss() for task in tasks.keys()}), requires_grad=False)
        self.test_acc = MultitaskAccuracy(tasks)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        batch_inputs, edges, edge_type, batch_labels, onset_divs, lengths = batch
        batch_size = lengths.shape[0]
        onset_edges = edges[:, edge_type == 0]
        onset_idx = unique_onsets(onset_divs)
        batch_pred = self.module((batch_inputs, edges, edge_type, onset_edges, onset_idx, lengths))
        # batch_pred = {k: v.reshape(-1, v.shape[-1]) for k, v in batch_pred.items()}
        batch_labels = {k: v.reshape(-1) for k, v in batch_labels.items()}
        loss = self.train_loss(batch_pred, batch_labels)
        self.log('train_loss', loss["total"].item(), on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        degree = torch.logical_and(
            batch_pred["degree1"].argmax(dim=1) == batch_labels["degree1"],
            batch_pred["degree2"].argmax(dim=1) == batch_labels["degree2"]).unsqueeze(0)
        root = (batch_pred["root"].argmax(dim=1) == batch_labels["root"]).unsqueeze(0)
        quality = (batch_pred["quality"].argmax(dim=1) == batch_labels["quality"]).unsqueeze(0)
        inversion = (batch_pred["inversion"].argmax(dim=1) == batch_labels["inversion"]).unsqueeze(0)
        local_key = (batch_pred["localkey"].argmax(dim=1) == batch_labels["localkey"]).unsqueeze(0)
        acc_RomNum = (torch.cat((degree, quality, root, inversion, local_key), dim=0).sum(0) == 5).float().mean()
        self.log('Train RomNum', acc_RomNum.item(), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("global_step", self.global_step, on_step=True, prog_bar=False, batch_size=batch_size)
        return loss["total"]

    def validation_step(self, batch, batch_idx):
        batch_inputs, edges, edge_type, batch_labels, onset_divs, name = batch
        onset_edges = edges[:, edge_type == 0]
        onset_idx = unique_onsets(onset_divs)
        batch_pred = self.module((batch_inputs, edges, edge_type, onset_edges, onset_idx, None))
        loss = self.val_loss(batch_pred, batch_labels)
        self.log('val_loss', loss["total"].item(), on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        degree = torch.logical_and(
            batch_pred["degree1"].argmax(dim=1) == batch_labels["degree1"],
            batch_pred["degree2"].argmax(dim=1) == batch_labels["degree2"]).unsqueeze(0)
        root = (batch_pred["root"].argmax(dim=1) == batch_labels["root"]).unsqueeze(0)
        quality = (batch_pred["quality"].argmax(dim=1) == batch_labels["quality"]).unsqueeze(0)
        inversion = (batch_pred["inversion"].argmax(dim=1) == batch_labels["inversion"]).unsqueeze(0)
        local_key = (batch_pred["localkey"].argmax(dim=1) == batch_labels["localkey"]).unsqueeze(0)
        acc_RomNum = (torch.cat((degree, quality, root, inversion, local_key), dim=0).sum(0) == 5).float()
        self.log('Val RomNum', acc_RomNum.mean().item(), on_step=False, on_epoch=True, prog_bar=True, batch_size=len(acc_RomNum))

    def test_step(self, batch, batch_idx):
        batch_inputs, edges, edge_type, batch_labels, onset_divs, name = batch
        onset_edges = edges[:, edge_type == 0]
        onset_idx = unique_onsets(onset_divs)
        batch_pred = self.module((batch_inputs, edges, edge_type, onset_edges, onset_idx, None))
        acc = self.test_acc(batch_pred, batch_labels)
        for task in self.tasks.keys():
            self.log(f'Test {task}', acc[task], on_epoch=True, prog_bar=True, on_step=False, batch_size=1)
        degree = torch.logical_and(
            batch_pred["degree1"].argmax(dim=1) == batch_labels["degree1"],
            batch_pred["degree2"].argmax(dim=1) == batch_labels["degree2"]).unsqueeze(0)
        root = (batch_pred["root"].argmax(dim=1) == batch_labels["root"]).unsqueeze(0)
        quality = (batch_pred["quality"].argmax(dim=1) == batch_labels["quality"]).unsqueeze(0)
        inversion = (batch_pred["inversion"].argmax(dim=1) == batch_labels["inversion"]).unsqueeze(0)
        local_key = (batch_pred["localkey"].argmax(dim=1) == batch_labels["localkey"]).unsqueeze(0)
        rn = (batch_pred["romanNumeral"].argmax(dim=1) == batch_labels["romanNumeral"]).unsqueeze(0)
        acc_RomNum = (torch.cat((degree, quality, root, inversion, local_key), dim=0).sum(0) == 5).float()
        acc_RomNumAlt = (torch.cat((rn, local_key, inversion), dim=0).sum(0) == 3).float()
        accs = {
            "Degree": degree.detach().cpu().squeeze().float(),
            "Root": root.detach().cpu().squeeze().float(),
            "Quality": quality.detach().cpu().squeeze().float(),
            "Inversion": inversion.detach().cpu().squeeze().float(),
            "Key": local_key.detach().cpu().squeeze().float(),
            "Roman Numeral": acc_RomNum.detach().cpu(),
            "Roman Numeral Alt": acc_RomNumAlt.detach().cpu()
        }
        if self.num_tasks == 14:
            # SATB Roman Numeral predictions can fully replace conventional Roman Numeral and improve performance.
            bass = (batch_pred["bass"].argmax(dim=1) == batch_labels["bass"]).unsqueeze(0)
            tenor = (batch_pred["tenor"].argmax(dim=1) == batch_labels["tenor"]).unsqueeze(0)
            alto = (batch_pred["alto"].argmax(dim=1) == batch_labels["alto"]).unsqueeze(0)
            soprano = (batch_pred["soprano"].argmax(dim=1) == batch_labels["soprano"]).unsqueeze(0)
            satb_rn = (torch.cat((bass, tenor, alto, soprano, local_key), dim=0).sum(0) == 5).float()
            accs["Roman Numeral"] = satb_rn.detach().cpu()

        acc_ts_RomNum = self.acc_compute_time_step(accs, batch_labels["onset"].detach().cpu())
        self.log('Test Degree', degree.float().mean().item(), on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        self.log('Test Roman Numeral (Onset)', acc_RomNum.mean().item(), on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        if isinstance(acc_ts_RomNum, dict):
            for k, v in acc_ts_RomNum.items():
                self.log(f'Test {k} CSR', v.mean().item(), on_step=False,
                         on_epoch=True, prog_bar=True, batch_size=1)
        else:
            self.log('Test Roman Numeral Time Step Accuracy', acc_ts_RomNum.mean().item(), on_step=False, on_epoch=True, prog_bar=True, batch_size=1)

    # def test_epoch_end(self, outputs):
    #     acc_RomNum = torch.cat([o[0] for o in outputs])
    #     acc_ts_RomNum = torch.cat([torch.tensor(o[1]) for o in outputs])
    #     self.log('Test Epoch RomNum', acc_RomNum.mean().item(), on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
    #     self.log('Test Epoch Roman Numeral Time Step Accuracy', acc_ts_RomNum.mean().item(), on_step=False, on_epoch=True, prog_bar=True, batch_size=1)

    def predict_step(self, batch, batch_idx):
        batch_inputs, edges, edge_type, batch_labels, name = batch
        onset_edges = edges[:, edge_type == 0]
        onset_idx = unique_onsets(batch_labels["onset"])
        batch_pred = self.module((batch_inputs, edges, edge_type, onset_edges, onset_idx))
        return batch_pred

    def acc_compute_time_step(self, acc_RomNum, onset):
        if isinstance(acc_RomNum, torch.Tensor):
            df = pd.DataFrame({"onset": onset, "acc": acc_RomNum})
        elif isinstance(acc_RomNum, dict):

            acc_RomNum["onset"] = onset
            df = pd.DataFrame(acc_RomNum)
        dfout = copy.deepcopy(df)
        dfout["onset"] = dfout["onset"] - dfout["onset"].min()
        for i in range(1, len(df)):
            onset_diff = int((df["onset"][i] - df["onset"][i - 1]) / 0.125) - 1
            row = df.iloc[i - 1]
            for j in range(onset_diff):
                row["onset"] = row["onset"] + 0.125
                dfout = dfout.append(row, ignore_index=True)
        dfout.sort_values(by="onset", inplace=True)
        if isinstance(acc_RomNum, dict):
            return {k: dfout[k].to_numpy() for k in acc_RomNum.keys()}
        else:
            return dfout["acc"].to_numpy()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
                {'params': self.module.parameters(), "lr": self.lr, "weight_decay": self.weight_decay},
                {'params': self.train_loss.parameters(), 'weight_decay': 0, "lr": self.lr}]
            )
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80], gamma=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }


class SingleTaskPrediction(LightningModule):
    def __init__(self,
                 in_feats: int,
                 n_hidden: int,
                 tasks: dict,
                 n_layers: int,
                 activation=F.relu,
                 dropout=0.5,
                 lr=0.0001,
                 weight_decay=5e-4,
                 use_nade=False,
                 use_jk=False,
                 use_rotograd=False,
                 device = 0
                 ):
        super(SingleTaskPrediction, self).__init__()
        self.tasks = tasks
        self.use_rotograd = use_rotograd
        self.save_hyperparameters()

        self.module = ChordPredictionModel(
            in_feats, n_hidden, tasks, n_layers, activation, dropout,
            use_nade=use_nade, use_jk=use_jk).float().to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_loss = MultiTaskLoss(
            list(tasks.keys()), nn.ModuleDict({task: nn.CrossEntropyLoss(ignore_index=-1) for task in tasks.keys()}), requires_grad=False)
        self.val_loss = MultiTaskLoss(
            list(tasks.keys()), nn.ModuleDict({task: nn.CrossEntropyLoss() for task in tasks.keys()}), requires_grad=False)
        self.test_acc = MultitaskAccuracy(tasks)

    def training_step(self, batch, batch_idx):
        batch_inputs, edges, edge_type, batch_labels, onset_divs, lengths = batch
        onset_edges = edges[:, edge_type == 0]
        onset_idx = unique_onsets(onset_divs)
        batch_pred = self.module((batch_inputs, edges, edge_type, onset_edges, onset_idx, lengths))
        batch_pred = {k: v.reshape(-1, v.shape[-1]) for k, v in batch_pred.items()}
        batch_labels = {k: v.reshape(-1) for k, v in batch_labels.items()}
        loss = self.train_loss(batch_pred, batch_labels)
        task = list(self.tasks.keys())[0]
        acc = (batch_pred[task].argmax(1) == batch_labels[task]).float().mean()
        self.log('train_loss', loss["total"].item(), on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        self.log(f'Train Accuracy', acc.item(), on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        return loss["total"]

    def validation_step(self, batch, batch_idx):
        batch_inputs, edges, edge_type, batch_labels, onset_divs, name = batch
        onset_edges = edges[:, edge_type == 0]
        onset_idx = unique_onsets(onset_divs)
        batch_pred = self.module((batch_inputs, edges, edge_type, onset_edges, onset_idx, None))
        loss = self.val_loss(batch_pred, batch_labels)
        self.log('val_loss', loss["total"].item(), on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        task = list(self.tasks.keys())[0]
        acc = (batch_pred[task].argmax(1) == batch_labels[task]).float().mean()
        self.log(f'Val Accuracy', acc.item(), on_step=False, on_epoch=True, prog_bar=True, batch_size=1)

    def test_step(self, batch, batch_idx):
        batch_inputs, edges, edge_type, batch_labels, onset_divs, name = batch
        onset_edges = edges[:, edge_type == 0]
        onset_idx = unique_onsets(onset_divs)
        batch_pred = self.module((batch_inputs, edges, edge_type, onset_edges, onset_idx, None))
        task = list(self.tasks.keys())[0]
        acc = (batch_pred[task].argmax(1) == batch_labels[task]).float().mean()
        self.log(f'Test Accuracy', acc.item(), on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        acc_ts_RomNum = self.acc_compute_time_step(acc.detach().cpu().numpy(), batch_labels["onset"].detach().cpu().numpy())
        self.log(f'Test Time Step Accuracy', acc_ts_RomNum.item(), on_step=False, on_epoch=True, prog_bar=True, batch_size=1)

    def predict_step(self, batch, batch_idx):
        batch_inputs, edges, edge_type, batch_labels, name = batch
        onset_edges = edges[:, edge_type == 0]
        batch_pred = self.module(batch_inputs, edges, edge_type, onset_edges)
        return batch_pred

    def acc_compute_time_step(self, acc_RomNum, onset):
        df = pd.DataFrame({"onset": onset, "acc": acc_RomNum})
        dfout = copy.deepcopy(df)
        dfout["onset"] = dfout["onset"] - dfout["onset"].min()
        for i in range(1, len(df)):
            onset_diff = int((df["onset"][i] - df["onset"][i - 1]) / 0.125) - 1
            row = df.iloc[i - 1]
            for j in range(onset_diff):
                row["onset"] = row["onset"] + 0.125
                dfout = dfout.append(row, ignore_index=True)
        dfout.sort_values(by="onset", inplace=True)
        return dfout["acc"].to_numpy().mean()

    def configure_optimizers(self):
        if self.use_rotograd:
            optimizer = torch.optim.AdamW(
                [{'params': m.parameters()} for m in self.module._backbone + self.module.heads] +\
                [{'params': self.module.parameters(), 'lr': self.lr}],
                lr=self.lr,
                weight_decay=self.weight_decay
            )
        else:
            optimizer = torch.optim.AdamW([
                {'params': self.module.parameters(), "lr": self.lr, "weight_decay": self.weight_decay},
                {'params': self.train_loss.parameters(), 'weight_decay': 0, "lr": self.lr}]
            )
        # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=5e-3, step_size_up=1000, cycle_momentum=False)
        return {
            "optimizer": optimizer,
            # "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }



class PostChordPrediction(LightningModule):
    def __init__(self,
                 in_feats: int,
                 n_hidden: int,
                 tasks: dict,
                 n_layers: int,
                 activation=F.relu,
                 dropout=0.5,
                 lr=0.0001,
                 weight_decay=5e-4,
                 use_nade=False,
                 use_jk=False,
                 use_rotograd=False,
                 device=0,
                 frozen_model = ChordPredictionModel(83)
                 ):
        super(PostChordPrediction, self).__init__()
        self.tasks = tasks
        self.num_tasks = len(tasks.keys())
        self.use_rotograd = use_rotograd
        self.save_hyperparameters()
        assert(frozen_model is not None, "Need a frozen model to initialize the post-processing module.")
        self.frozen_model = frozen_model
        self.module = PostProcessingMLTModel(tasks, n_hidden, n_layers, activation, dropout).float().to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.test_roman = list()
        self.test_roman_ts = list()
        self.train_loss = MultiTaskLoss(
            list(tasks.keys()), nn.ModuleDict({task: nn.CrossEntropyLoss(ignore_index=-1) for task in tasks.keys()}), requires_grad=not use_rotograd)
        self.val_loss = MultiTaskLoss(
            list(tasks.keys()), nn.ModuleDict({task: nn.CrossEntropyLoss() for task in tasks.keys()}), requires_grad=False)
        self.test_acc = MultitaskAccuracy(tasks)

    def training_step(self, batch, batch_idx):
        batch_inputs, edges, edge_type, batch_labels, onset_divs, lengths = batch
        batch_size = lengths.shape[0]
        onset_edges = edges[:, edge_type == 0]
        onset_idx = unique_onsets(onset_divs)
        b_size = batch_labels["localkey"].shape[-1]
        with torch.no_grad():
            self.frozen_model.eval()
            batch_inputs = self.frozen_model((batch_inputs, edges, edge_type, onset_edges, onset_idx, lengths))
        batch_inputs = {k: v.reshape(-1, b_size, v.shape[-1]) for k, v in batch_inputs.items()}
        batch_pred = self.module(batch_inputs)
        batch_pred = {k: v.reshape(-1, v.shape[-1]) for k, v in batch_pred.items()}
        batch_labels = {k: v.reshape(-1) for k, v in batch_labels.items()}
        loss = self.train_loss(batch_pred, batch_labels)
        self.log('train_loss', loss["total"].item(), on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        degree = torch.logical_and(
            batch_pred["degree1"].argmax(dim=1) == batch_labels["degree1"],
            batch_pred["degree2"].argmax(dim=1) == batch_labels["degree2"]).unsqueeze(0)
        root = (batch_pred["root"].argmax(dim=1) == batch_labels["root"]).unsqueeze(0)
        quality = (batch_pred["quality"].argmax(dim=1) == batch_labels["quality"]).unsqueeze(0)
        inversion = (batch_pred["inversion"].argmax(dim=1) == batch_labels["inversion"]).unsqueeze(0)
        local_key = (batch_pred["localkey"].argmax(dim=1) == batch_labels["localkey"]).unsqueeze(0)
        acc_RomNum = (torch.cat((degree, quality, root, inversion, local_key), dim=0).sum(0) == 5).float().mean()
        self.log('Train RomNum', acc_RomNum.item(), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        return loss["total"]

    def validation_step(self, batch, batch_idx):
        batch_inputs, edges, edge_type, batch_labels, onset_divs, name = batch
        onset_edges = edges[:, edge_type == 0]
        onset_idx = unique_onsets(onset_divs)
        batch_inputs = self.frozen_model((batch_inputs, edges, edge_type, onset_edges, onset_idx, None))
        batch_pred = self.module(batch_inputs)
        loss = self.val_loss(batch_pred, batch_labels)
        self.log('val_loss', loss["total"].item(), on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        degree = torch.logical_and(
            batch_pred["degree1"].argmax(dim=1) == batch_labels["degree1"],
            batch_pred["degree2"].argmax(dim=1) == batch_labels["degree2"]).unsqueeze(0)
        root = (batch_pred["root"].argmax(dim=1) == batch_labels["root"]).unsqueeze(0)
        quality = (batch_pred["quality"].argmax(dim=1) == batch_labels["quality"]).unsqueeze(0)
        inversion = (batch_pred["inversion"].argmax(dim=1) == batch_labels["inversion"]).unsqueeze(0)
        local_key = (batch_pred["localkey"].argmax(dim=1) == batch_labels["localkey"]).unsqueeze(0)
        acc_RomNum = (torch.cat((degree, quality, root, inversion, local_key), dim=0).sum(0) == 5).float()
        self.log('Val RomNum', acc_RomNum.mean().item(), on_step=False, on_epoch=True, prog_bar=True, batch_size=len(acc_RomNum))

    def test_step(self, batch, batch_idx):
        batch_inputs, edges, edge_type, batch_labels, onset_divs, name = batch
        onset_edges = edges[:, edge_type == 0]
        onset_idx = unique_onsets(onset_divs)
        batch_inputs = self.frozen_model((batch_inputs, edges, edge_type, onset_edges, onset_idx, None))
        batch_pred = self.module(batch_inputs)
        acc = self.test_acc(batch_pred, batch_labels)
        for task in self.tasks.keys():
            self.log(f'Test {task}', acc[task], on_epoch=True, prog_bar=True, on_step=False, batch_size=1)
        degree = torch.logical_and(
            batch_pred["degree1"].argmax(dim=1) == batch_labels["degree1"],
            batch_pred["degree2"].argmax(dim=1) == batch_labels["degree2"]).unsqueeze(0)
        root = (batch_pred["root"].argmax(dim=1) == batch_labels["root"]).unsqueeze(0)
        quality = (batch_pred["quality"].argmax(dim=1) == batch_labels["quality"]).unsqueeze(0)
        inversion = (batch_pred["inversion"].argmax(dim=1) == batch_labels["inversion"]).unsqueeze(0)
        local_key = (batch_pred["localkey"].argmax(dim=1) == batch_labels["localkey"]).unsqueeze(0)
        rn = (batch_pred["romanNumeral"].argmax(dim=1) == batch_labels["romanNumeral"]).unsqueeze(0)
        acc_RomNum = (torch.cat((degree, quality, root, inversion, local_key), dim=0).sum(0) == 5).float()
        acc_RomNumAlt = (torch.cat((rn, local_key, inversion), dim=0).sum(0) == 3).float()
        accs = {
            "Degree": degree.detach().cpu().squeeze().float(),
            "Root": root.detach().cpu().squeeze().float(),
            "Quality": quality.detach().cpu().squeeze().float(),
            "Inversion": inversion.detach().cpu().squeeze().float(),
            "Key": local_key.detach().cpu().squeeze().float(),
            "Roman Numeral": acc_RomNum.detach().cpu(),
            "Roman Numeral Alt": acc_RomNumAlt.detach().cpu()
        }
        if self.num_tasks == 14:
            # SATB Roman Numeral predictions can fully replace conventional Roman Numeral and improve performance.
            bass = (batch_pred["bass"].argmax(dim=1) == batch_labels["bass"]).unsqueeze(0)
            tenor = (batch_pred["tenor"].argmax(dim=1) == batch_labels["tenor"]).unsqueeze(0)
            alto = (batch_pred["alto"].argmax(dim=1) == batch_labels["alto"]).unsqueeze(0)
            soprano = (batch_pred["soprano"].argmax(dim=1) == batch_labels["soprano"]).unsqueeze(0)
            satb_rn = (torch.cat((bass, tenor, alto, soprano, local_key), dim=0).sum(0) == 5).float()
            accs["Roman Numeral"] = satb_rn.detach().cpu()
        acc_ts_RomNum = self.acc_compute_time_step(accs, batch_labels["onset"].detach().cpu())
        self.log('Test Degree', degree.float().mean().item(), on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        self.log('Test Roman Numeral (Onset)', acc_RomNum.mean().item(), on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        if isinstance(acc_ts_RomNum, dict):
            for k, v in acc_ts_RomNum.items():
                self.log(f'Test {k} CSR', v.mean().item(), on_step=False,
                         on_epoch=True, prog_bar=True, batch_size=1)
        else:
            self.log('Test Roman Numeral Time Step Accuracy', acc_ts_RomNum.mean().item(), on_step=False, on_epoch=True, prog_bar=True, batch_size=1)

    def predict_step(self, batch, batch_idx):
        batch_inputs, edges, edge_type, batch_labels, name = batch
        onset_edges = edges[:, edge_type == 0]
        onset_idx = unique_onsets(batch_labels["onset"])
        batch_pred = self.module((batch_inputs, edges, edge_type, onset_edges, onset_idx))
        return batch_pred

    def acc_compute_time_step(self, acc_RomNum, onset):
        if isinstance(acc_RomNum, torch.Tensor):
            df = pd.DataFrame({"onset": onset, "acc": acc_RomNum})
        elif isinstance(acc_RomNum, dict):

            acc_RomNum["onset"] = onset
            df = pd.DataFrame(acc_RomNum)
        dfout = copy.deepcopy(df)
        dfout["onset"] = dfout["onset"] - dfout["onset"].min()
        for i in range(1, len(df)):
            onset_diff = int((df["onset"][i] - df["onset"][i - 1]) / 0.125) - 1
            row = df.iloc[i - 1]
            for j in range(onset_diff):
                row["onset"] = row["onset"] + 0.125
                dfout = dfout.append(row, ignore_index=True)
        dfout.sort_values(by="onset", inplace=True)
        if isinstance(acc_RomNum, dict):
            return {k: dfout[k].to_numpy() for k in acc_RomNum.keys()}
        else:
            return dfout["acc"].to_numpy()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
                {'params': self.module.parameters(), "lr": self.lr, "weight_decay": self.weight_decay},
                {'params': self.train_loss.parameters(), 'weight_decay': 0, "lr": self.lr}]
            )
        return {
            "optimizer": optimizer,
            "monitor": "val_loss"
        }


class MetricalChordPrediction(LightningModule):
    def __init__(self,
                 in_feats: int,
                 n_hidden: int,
                 tasks: dict,
                 n_layers: int,
                 activation=F.relu,
                 dropout=0.5,
                 lr=0.0001,
                 weight_decay=5e-4,
                 use_nade=False,
                 use_jk=False,
                 weight_loss=True,
                 use_metrical=False,
                 use_reledge=False,
                 **kwargs
                 ):
        super(MetricalChordPrediction, self).__init__()
        self.tasks = tasks
        self.save_hyperparameters()
        self.num_tasks = len(tasks.keys())
        self.use_reledge = use_reledge
        self.etypes = {"onset": 0, "consecutive": 1, "during": 2, "rest": 3, "consecutive_rev": 4, "during_rev": 5,
                       "rest_rev": 6}
        self.use_signed_features = kwargs.get("use_signed_features", False)
        self.module = MetricalChordPredictionModel(
            in_feats, n_hidden, tasks, n_layers, activation, dropout,
            use_nade=use_nade, use_jk=use_jk, use_reledge=use_reledge, metrical=use_metrical, **kwargs).float().to(self.device)
        pitch_embedding = kwargs.get("pitch_embedding", None)
        self.pitch_embedding = torch.nn.Embedding(12, 16) if pitch_embedding is not None else pitch_embedding
        self.lr = lr
        self.weight_decay = weight_decay
        self.test_roman = list()
        self.test_roman_ts = list()
        self.train_loss = MultiTaskLoss(
            list(tasks.keys()), nn.ModuleDict({task: nn.CrossEntropyLoss(ignore_index=-1, label_smoothing=0.1) for task in tasks.keys()}), requires_grad=weight_loss)
        self.val_loss = MultiTaskLoss(
            list(tasks.keys()), nn.ModuleDict({task: nn.CrossEntropyLoss() for task in tasks.keys()}), requires_grad=False)
        self.test_acc = MultitaskAccuracy(tasks)

    def training_step(self, batch, batch_idx):
        batch_size = batch["lengths"].shape[0]
        onset_edges = batch["edge_index"][:, batch["edge_type"] == 0]
        onset_idx = unique_onsets(batch["onset_div"])
        # Add reverse edges
        edges, edge_types = add_reverse_edges_from_edge_index(batch["edge_index"], batch["edge_type"])
        na = batch["note_array"]
        if self.use_reledge:
            edge_features = batch["note_array"][edges[1]] - batch["note_array"][edges[0]]
            edge_features = edge_features if self.use_signed_features else torch.abs(edge_features)
            edge_features = F.normalize(edge_features, dim=0)
        else:
            edge_features = None
        if self.pitch_embedding is not None and edge_features is not None:
            pitch = self.pitch_embedding(torch.remainder(na[:, 0][edges[0]] - na[:, 0][edges[1]], 12).long())
            edge_features = torch.cat([edge_features, pitch], dim=1)

        x_dict = {"note": batch["x"]}
        edge_index_dict = {et: edges[:, edge_types == self.etypes[et[1]]] for et in METADATA[1]}
        edge_feature_dict = {et: edge_features[edge_types == self.etypes[et[1]]] for et in
                             METADATA[1]} if edge_features is not None else None
        batch_pred = self.module(x_dict, edge_index_dict, edge_feature_dict, onset_edges, onset_idx,
                                 lengths=batch["lengths"])

        batch_labels = {k: v.reshape(-1) for k, v in batch["y"].items()}
        loss = self.train_loss(batch_pred, batch_labels)
        self.log('train_loss', loss["total"].item(), on_step=False, on_epoch=True, prog_bar=False, batch_size=batch_size)
        degree = torch.logical_and(
            batch_pred["degree1"].argmax(dim=1) == batch_labels["degree1"],
            batch_pred["degree2"].argmax(dim=1) == batch_labels["degree2"]).unsqueeze(0)
        root = (batch_pred["root"].argmax(dim=1) == batch_labels["root"]).unsqueeze(0)
        quality = (batch_pred["quality"].argmax(dim=1) == batch_labels["quality"]).unsqueeze(0)
        inversion = (batch_pred["inversion"].argmax(dim=1) == batch_labels["inversion"]).unsqueeze(0)
        local_key = (batch_pred["localkey"].argmax(dim=1) == batch_labels["localkey"]).unsqueeze(0)
        acc_RomNum = (torch.cat((degree, quality, root, inversion, local_key), dim=0).sum(0) == 5).float().mean()
        self.log('Train RomNum', acc_RomNum.item(), on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("global_step", self.global_step, on_step=True, prog_bar=False, batch_size=batch_size)
        return loss["total"]

    def validation_step(self, batch, batch_idx):
        onset_edges = batch["edge_index"][:, batch["edge_type"] == 0]
        onset_idx = unique_onsets(batch["onset_div"])
        # Add reverse edges
        edges, edge_types = add_reverse_edges_from_edge_index(batch["edge_index"], batch["edge_type"])
        na = batch["note_array"]
        if self.use_reledge:
            edge_features = batch["note_array"][edges[1]] - batch["note_array"][edges[0]]
            edge_features = edge_features if self.use_signed_features else torch.abs(edge_features)
            edge_features = F.normalize(edge_features, dim=0)
        else:
            edge_features = None
        if self.pitch_embedding is not None and edge_features is not None:
            pitch = self.pitch_embedding(torch.remainder(na[:, 0][edges[0]] - na[:, 0][edges[1]], 12).long())
            edge_features = torch.cat([edge_features, pitch], dim=1)
        x_dict = {"note": batch["x"]}
        edge_index_dict = {et: edges[:, edge_types == self.etypes[et[1]]] for et in METADATA[1]}
        edge_feature_dict = {et: edge_features[edge_types == self.etypes[et[1]]] for et in
                             METADATA[1]} if edge_features is not None else None
        batch_pred = self.module(x_dict, edge_index_dict, edge_feature_dict, onset_edges, onset_idx,
                                 lengths=batch["lengths"])
        batch_labels = batch["y"]
        loss = self.val_loss(batch_pred, batch_labels)
        self.log('val_loss', loss["total"].item(), on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        degree = torch.logical_and(
            batch_pred["degree1"].argmax(dim=1) == batch_labels["degree1"],
            batch_pred["degree2"].argmax(dim=1) == batch_labels["degree2"]).unsqueeze(0)
        root = (batch_pred["root"].argmax(dim=1) == batch_labels["root"]).unsqueeze(0)
        quality = (batch_pred["quality"].argmax(dim=1) == batch_labels["quality"]).unsqueeze(0)
        inversion = (batch_pred["inversion"].argmax(dim=1) == batch_labels["inversion"]).unsqueeze(0)
        local_key = (batch_pred["localkey"].argmax(dim=1) == batch_labels["localkey"]).unsqueeze(0)
        acc_RomNum = (torch.cat((degree, quality, root, inversion, local_key), dim=0).sum(0) == 5).float()
        self.log('Val RomNum', acc_RomNum.mean().item(), on_step=False, on_epoch=True, prog_bar=True, batch_size=len(acc_RomNum))

    def test_step(self, batch, batch_idx):
        onset_edges = batch["edge_index"][:, batch["edge_type"] == 0]
        onset_idx = unique_onsets(batch["onset_div"])
        # Add reverse edges
        edges, edge_types = add_reverse_edges_from_edge_index(batch["edge_index"], batch["edge_type"])
        na = batch["note_array"]
        if self.use_reledge:
            edge_features = batch["note_array"][edges[1]] - batch["note_array"][edges[0]]
            edge_features = edge_features if self.use_signed_features else torch.abs(edge_features)
            edge_features = F.normalize(edge_features, dim=0)
        else:
            edge_features = None
        if self.pitch_embedding is not None and edge_features is not None:
            pitch = self.pitch_embedding(torch.remainder(na[:, 0][edges[0]] - na[:, 0][edges[1]], 12).long())
            edge_features = torch.cat([edge_features, pitch], dim=1)
        x_dict = {"note": batch["x"]}
        edge_index_dict = {et: edges[:, edge_types == self.etypes[et[1]]] for et in METADATA[1]}
        edge_feature_dict = {et: edge_features[edge_types == self.etypes[et[1]]] for et in
                             METADATA[1]} if edge_features is not None else None
        batch_pred = self.module(x_dict, edge_index_dict, edge_feature_dict, onset_edges, onset_idx,
                                 lengths=batch["lengths"])
        batch_labels = batch["y"]
        acc = self.test_acc(batch_pred, batch_labels)
        for task in self.tasks.keys():
            self.log(f'Test {task}', acc[task], on_epoch=True, prog_bar=True, on_step=False, batch_size=1)
        degree = torch.logical_and(
            batch_pred["degree1"].argmax(dim=1) == batch_labels["degree1"],
            batch_pred["degree2"].argmax(dim=1) == batch_labels["degree2"]).unsqueeze(0)
        root = (batch_pred["root"].argmax(dim=1) == batch_labels["root"]).unsqueeze(0)
        quality = (batch_pred["quality"].argmax(dim=1) == batch_labels["quality"]).unsqueeze(0)
        inversion = (batch_pred["inversion"].argmax(dim=1) == batch_labels["inversion"]).unsqueeze(0)
        local_key = (batch_pred["localkey"].argmax(dim=1) == batch_labels["localkey"]).unsqueeze(0)
        rn = (batch_pred["romanNumeral"].argmax(dim=1) == batch_labels["romanNumeral"]).unsqueeze(0)
        acc_RomNum = (torch.cat((degree, quality, root, inversion, local_key), dim=0).sum(0) == 5).float()
        acc_RomNumAlt = (torch.cat((rn, local_key, inversion), dim=0).sum(0) == 3).float()
        accs = {
            "Degree": degree.detach().cpu().squeeze().float(),
            "Root": root.detach().cpu().squeeze().float(),
            "Quality": quality.detach().cpu().squeeze().float(),
            "Inversion": inversion.detach().cpu().squeeze().float(),
            "Key": local_key.detach().cpu().squeeze().float(),
            "Roman Numeral": acc_RomNum.detach().cpu(),
            "Roman Numeral Alt": acc_RomNumAlt.detach().cpu()
        }
        if self.num_tasks == 14:
            # SATB Roman Numeral predictions can fully replace conventional Roman Numeral and improve performance.
            bass = (batch_pred["bass"].argmax(dim=1) == batch_labels["bass"]).unsqueeze(0)
            tenor = (batch_pred["tenor"].argmax(dim=1) == batch_labels["tenor"]).unsqueeze(0)
            alto = (batch_pred["alto"].argmax(dim=1) == batch_labels["alto"]).unsqueeze(0)
            soprano = (batch_pred["soprano"].argmax(dim=1) == batch_labels["soprano"]).unsqueeze(0)
            satb_rn = (torch.cat((bass, tenor, alto, soprano, local_key), dim=0).sum(0) == 5).float()
            accs["Roman Numeral"] = satb_rn.detach().cpu()

        acc_ts_RomNum = self.acc_compute_time_step(accs, batch_labels["onset"].detach().cpu())
        self.log('Test Degree', degree.float().mean().item(), on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        self.log('Test Roman Numeral (Onset)', acc_RomNum.mean().item(), on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
        if isinstance(acc_ts_RomNum, dict):
            for k, v in acc_ts_RomNum.items():
                self.log(f'Test {k} CSR', v.mean().item(), on_step=False,
                         on_epoch=True, prog_bar=True, batch_size=1)
        else:
            self.log('Test Roman Numeral Time Step Accuracy', acc_ts_RomNum.mean().item(), on_step=False, on_epoch=True, prog_bar=True, batch_size=1)

    # def test_epoch_end(self, outputs):
    #     acc_RomNum = torch.cat([o[0] for o in outputs])
    #     acc_ts_RomNum = torch.cat([torch.tensor(o[1]) for o in outputs])
    #     self.log('Test Epoch RomNum', acc_RomNum.mean().item(), on_step=False, on_epoch=True, prog_bar=True, batch_size=1)
    #     self.log('Test Epoch Roman Numeral Time Step Accuracy', acc_ts_RomNum.mean().item(), on_step=False, on_epoch=True, prog_bar=True, batch_size=1)

    def predict_step(self, batch, batch_idx):
        batch_inputs, edges, edge_type, batch_labels, name = batch
        onset_edges = edges[:, edge_type == 0]
        onset_idx = unique_onsets(batch_labels["onset"])
        batch_pred = self.module((batch_inputs, edges, edge_type, onset_edges, onset_idx))
        return batch_pred

    def acc_compute_time_step(self, acc_RomNum, onset):
        if isinstance(acc_RomNum, torch.Tensor):
            df = pd.DataFrame({"onset": onset, "acc": acc_RomNum})
        elif isinstance(acc_RomNum, dict):

            acc_RomNum["onset"] = onset
            df = pd.DataFrame(acc_RomNum)
        dfout = copy.deepcopy(df)
        dfout["onset"] = dfout["onset"] - dfout["onset"].min()
        for i in range(1, len(df)):
            onset_diff = int((df["onset"][i] - df["onset"][i - 1]) / 0.125) - 1
            row = df.iloc[i - 1]
            for j in range(onset_diff):
                row["onset"] = row["onset"] + 0.125
                dfout = dfout.append(row, ignore_index=True)
        dfout.sort_values(by="onset", inplace=True)
        if isinstance(acc_RomNum, dict):
            return {k: dfout[k].to_numpy() for k in acc_RomNum.keys()}
        else:
            return dfout["acc"].to_numpy()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {'params': self.module.parameters(), "lr": self.lr, "weight_decay": self.weight_decay},
            {'params': self.train_loss.parameters(), 'weight_decay': 0, "lr": self.lr}]
        )
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80], gamma=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }


def unique_onsets(onsets):
    unique, inverse = torch.unique(onsets, sorted=True, return_inverse=True)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
    return perm