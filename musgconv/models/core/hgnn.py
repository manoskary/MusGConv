import torch.nn as nn
from torch.nn import functional as F
import torch
from .gnn import SageConvScatter as SageConv, ResGatedGraphConv, JumpingKnowledge, RelEdgeConv, MetricalConvLayer, MusGConv
from torch_scatter import scatter_add
from torch_geometric.nn import to_hetero


class HeteroAttention(nn.Module):
    def __init__(self, n_hidden, n_layers):
        super(HeteroAttention, self).__init__()
        self.lstm = nn.LSTM(n_hidden, (n_layers*n_hidden)//2, bidirectional=True, batch_first=True)
        self.att = nn.Linear(2 * ((n_layers*n_hidden)//2), 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()
        nn.init.xavier_uniform_(self.att.weight, gain=nn.init.calculate_gain("relu"))

    def forward(self, x):
        alpha, _ = self.lstm(x)
        alpha = self.att(alpha).squeeze(-1)  # [num_nodes, num_layers]
        alpha = torch.softmax(alpha, dim=-1)
        return (x * alpha.unsqueeze(-1)).sum(dim=0)


class HeteroResGatedGraphConvLayer(nn.Module):
    def __init__(self, in_features, out_features, etypes, bias=True, reduction='mean'):
        super(HeteroResGatedGraphConvLayer, self).__init__()
        self.out_features = out_features
        self.etypes = etypes
        if reduction == 'mean':
            self.reduction = lambda x: x.mean(dim=0)
        elif reduction == 'sum':
            self.reduction = lambda x: x.sum(dim=0)
        elif reduction == 'max':
            self.reduction = lambda x: x.max(dim=0)
        elif reduction == 'min':
            self.reduction = lambda x: x.min(dim=0)
        elif reduction == 'concat':
            self.reduction = lambda x: torch.cat(x, dim=0)
        elif reduction == 'lstm':
            self.reduction = HeteroAttention(out_features, len(etypes.keys()))
        elif reduction == 'none':
            self.reduction = lambda x: x
        else:
            raise NotImplementedError

        conv_dict = dict()
        for etype in etypes.keys():
            conv_dict[etype] = ResGatedGraphConv(in_features, out_features, bias=bias)
        self.conv = nn.ModuleDict(conv_dict)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.conv.values():
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_type):
        out = torch.zeros((len(self.conv.keys()), x.shape[0], self.out_features))
        for idx, (ekey, evalue) in enumerate(self.etypes.items()):
            mask = edge_type == evalue
            out[idx] = self.conv[ekey](x, edge_index[:, mask])
        return self.reduction(out).to(x.device)


class HeteroRelEdgeConvLayer(nn.Module):
    def __init__(self, in_features, out_features, etypes, bias=True, in_edge_features=None):
        super(HeteroRelEdgeConvLayer, self).__init__()
        self.out_features = out_features
        self.etypes = etypes
        conv_dict = dict()
        for etype in etypes.keys():
            conv_dict[etype] = RelEdgeConv(in_features, out_features, bias=bias, in_edge_features=in_edge_features)
        self.conv = nn.ModuleDict(conv_dict)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.conv.values():
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_type, edge_features=None):
        if edge_features is not None and edge_features.shape[0] == x.shape[0]:
            edge_features = torch.abs(edge_features[edge_index[0]] - edge_features[edge_index[1]])
        elif edge_features is not None and edge_features.shape[0] == edge_index.shape[1]:
            edge_features = edge_features
        else:
            edge_features = None
        out = torch.zeros((len(self.conv.keys()), x.shape[0], self.out_features))
        for idx, (ekey, evalue) in enumerate(self.etypes.items()):
            mask = edge_type == evalue
            if edge_features is None:
                out[idx] = self.conv[ekey](x, edge_index[:, mask])
            else:
                out[idx] = self.conv[ekey](x, edge_index[:, mask], edge_features[mask])
        return out.mean(dim=0).to(x.device)


class HeteroSageConvLayer(nn.Module):
    def __init__(self, in_features, out_features, etypes, bias=True, reduction='mean'):
        super(HeteroSageConvLayer, self).__init__()
        self.out_features = out_features
        self.etypes = etypes
        if reduction == 'mean':
            self.reduction = lambda x: x.mean(dim=0)
        elif reduction == 'sum':
            self.reduction = lambda x: x.sum(dim=0)
        elif reduction == 'max':
            self.reduction = lambda x: x.max(dim=0)
        elif reduction == 'min':
            self.reduction = lambda x: x.min(dim=0)
        elif reduction == 'concat':
            self.reduction = lambda x: torch.cat(x, dim=0)
        elif reduction == 'lstm':
            self.reduction = HeteroAttention(out_features, len(etypes.keys()))
        else:
            raise NotImplementedError

        conv_dict = dict()
        for etype in etypes.keys():
            conv_dict[etype] = SageConv(in_features, out_features, bias=bias)
        self.conv = nn.ModuleDict(conv_dict)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.conv.values():
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_type):
        out = torch.zeros((len(self.conv.keys()), x.shape[0], self.out_features))
        for idx, (ekey, evalue) in enumerate(self.etypes.items()):
            mask = edge_type == evalue
            out[idx] = self.conv[ekey](x, edge_index[:, mask])
        return self.reduction(out).to(x.device)



class HGCN(nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats, n_layers, etypes={"onset":0, "consecutive":1, "during":2, "rest":3, "consecutive_rev":4, "during_rev":5, "rest_rev":6}, activation=F.relu, dropout=0.5, jk=False):
        super(HGCN, self).__init__()
        self.n_hidden = n_hidden
        self.layers = nn.ModuleList()
        self.normalize = F.normalize
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.layers.append(HeteroSageConvLayer(in_feats, n_hidden, etypes=etypes))
        for i in range(n_layers - 1):
            self.layers.append(HeteroSageConvLayer(n_hidden, n_hidden, etypes=etypes))
        if jk:
            self.use_knowledge = True
            self.jk = JumpingKnowledge(n_hidden=n_hidden, n_layers=n_layers)
        else:
            self.use_knowledge = False
        self.layers.append(HeteroSageConvLayer(n_hidden, out_feats, etypes=etypes))
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.layers:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_type):
        h = x
        hs = []
        for conv in self.layers[:-1]:
            h = conv(h, edge_index, edge_type)
            h = self.activation(h)
            h = self.normalize(h)
            h = self.dropout(h)
            hs.append(h)
        if self.use_knowledge:
            h = self.jk(hs)
        h = self.layers[-1](h, edge_index, edge_type)
        return h


class HResGatedConv(nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats, n_layers, etypes={"onset":0, "consecutive":1, "during":2, "rest":3, "consecutive_rev":4, "during_rev":5, "rest_rev":6}, activation=F.relu, dropout=0.5, jk=False):
        super(HResGatedConv, self).__init__()
        self.n_hidden = n_hidden
        self.layers = nn.ModuleList()
        self.normalize = F.normalize
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.layers.append(HeteroResGatedGraphConvLayer(in_feats, n_hidden, etypes=etypes))
        for i in range(n_layers - 1):
            self.layers.append(HeteroResGatedGraphConvLayer(n_hidden, n_hidden, etypes=etypes))
        if jk:
            self.use_knowledge = True
            self.jk = JumpingKnowledge(n_hidden=n_hidden, n_layers=n_layers)
        else:
            self.use_knowledge = False
        self.layers.append(HeteroResGatedGraphConvLayer(n_hidden, out_feats, etypes=etypes))
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.layers:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_type):
        h = x
        hs = []
        for conv in self.layers:
            h = conv(h, edge_index, edge_type)
            h = self.activation(h)
            h = self.normalize(h)
            h = self.dropout(h)
            hs.append(h)
        if self.use_knowledge:
            h = self.jk(hs)
        # h = self.layers[-1](h, edge_index, edge_type)
        return h


class HGPSLayer(nn.Module):
    def __init__(
            self, in_features, out_features, num_heads,
            etypes={"onset":0, "consecutive":1, "during":2, "rest":3, "consecutive_rev":4, "during_rev":5, "rest_rev":6},
            activation=F.relu, dropout=0.2, bias=True):
        """
        General Powerful Scalable Graph Transformers Convolutional Layer

        Parameters
        ----------
        in_features: int
            Number of input features
        out_features: int
            Number of output features
        num_heads: int
            Number of attention heads
        etypes: dict
            Edge types
        activation: nn.Module
            Activation function
        dropout: float
            Dropout rate
        bias: bool
            Whether to use bias
        """
        super(HGPSLayer, self).__init__()
        self.embedding = nn.Linear(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.activation = activation
        self.normalize_local = nn.LayerNorm(out_features)
        self.normalize_attn = nn.LayerNorm(out_features)
        self.dropout_ff = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_local = nn.Dropout(dropout)
        self.ff1 = nn.Linear(out_features, out_features*2, bias=bias)
        self.ff2 = nn.Linear(out_features*2, out_features, bias=bias)
        self.attn = nn.MultiheadAttention(out_features, num_heads, dropout=dropout, bias=bias, batch_first=True)
        self.local = HeteroResGatedGraphConvLayer(out_features, out_features, bias=bias, etypes=etypes)

    def forward(self, x, edge_index, edge_type):

        h_init = self.embedding(x)
        # Local embeddings
        local_out = self.local(h_init, edge_index, edge_type)
        local_out = self.activation(local_out)
        local_out = self.normalize_local(local_out)
        local_out = self.dropout_local(local_out)
        local_out = local_out + h_init

        # Global embeddings
        h = h_init.unsqueeze(0)
        attn_out, _ = self.attn(h, h, h)
        attn_out = self.activation(attn_out)
        attn_out = self.normalize_attn(attn_out)
        attn_out = self.dropout_attn(attn_out)
        attn_out = attn_out + h_init

        # Combine
        out = local_out + attn_out.squeeze()
        h = self.ff1(out)
        h = self.activation(h)
        h = self.dropout_ff(h)
        h = self.ff2(h)
        h - self.dropout_ff(h)
        out = F.normalize(out + h)
        return out


class HGPS(nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats, n_layers, etypes={"onset":0, "consecutive":1, "during":2, "rest":3, "consecutive_rev":4, "during_rev":5, "rest_rev":6}, activation=F.relu, dropout=0.5, jk=False):
        super(HGPS, self).__init__()
        self.n_hidden = n_hidden
        self.layers = nn.ModuleList()
        self.normalize = F.normalize
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.layers.append(HGPSLayer(in_feats, n_hidden, etypes=etypes, num_heads=4))
        for i in range(n_layers - 1):
            self.layers.append(HGPSLayer(n_hidden, n_hidden, etypes=etypes, num_heads=4))
        if jk:
            self.use_knowledge = True
            self.jk = JumpingKnowledge(n_hidden=n_hidden, n_layers=n_layers)
        else:
            self.use_knowledge = False
        self.layers.append(HGPSLayer(n_hidden, out_feats, etypes=etypes, num_heads=4))

    def forward(self, x, edge_index, edge_type):
        h = x
        hs = []
        for conv in self.layers:
            h = conv(h, edge_index, edge_type)
            h = self.activation(h)
            h = self.normalize(h)
            h = self.dropout(h)
            hs.append(h)
        if self.use_knowledge:
            h = self.jk(hs)
        # h = self.layers[-1](h, edge_index, edge_type)
        return h


class MetricalGNN(nn.Module):
    def __init__(self, input_features, hidden_features, output_features, etypes, num_layers=2, dropout=0.5,
                 use_reledge=False, jk=False, in_edge_features:int=None, metrical=False, conv_block=SageConv,
                 stack_convs=False, **kwargs):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers
        self.num_hidden = hidden_features
        self.convs = nn.ModuleList()
        self.emb_beats = nn.Linear(input_features, hidden_features)
        self.emb_measures = nn.Linear(input_features, hidden_features)
        self.beat_convs = nn.ModuleList()
        self.measure_convs = nn.ModuleList()
        self.project_metrical = nn.ModuleList()
        self.use_reledge = use_reledge
        self.use_metrical = metrical
        self.is_heterogeneous = kwargs.get("hetero", True)
        in_edge_features = in_edge_features if use_reledge else None
        stack_features = in_edge_features if stack_convs and use_reledge else None
        jk = kwargs.get("use_jk", jk)
        if jk:
            self.use_knowledge = True
            self.jk = JumpingKnowledge(n_hidden=hidden_features, n_layers=hidden_features)
        else:
            self.use_knowledge = False

        if self.is_heterogeneous:
            self.convs.append(HeteroConv(input_features, hidden_features, etypes=etypes,
                                     in_edge_features=in_edge_features, module=conv_block))
        else:
            self.convs.append(conv_block(input_features, hidden_features, bias=True, in_edge_features=in_edge_features))
        if num_layers > 2:
            for _ in range(num_layers - 2):
                if self.is_heterogeneous:
                    self.convs.append(HeteroConv(hidden_features, hidden_features, etypes=etypes,
                                                 in_edge_features=stack_features, module=conv_block))
                else:
                    self.convs.append(conv_block(hidden_features, hidden_features, in_edge_features=stack_features,
                                                 bias=True))
        if self.is_heterogeneous:
            self.convs.append(to_hetero(hidden_features, hidden_features, etypes=etypes,
                                         in_edge_features=stack_features, module=conv_block))
        else:
            self.convs.append(conv_block(hidden_features, hidden_features, bias=True, in_edge_features=stack_features))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_type, beat_nodes, measure_nodes, beat_edges, measure_edges,
                rel_edge=None, beat_lengths=None, measure_lengths=None, **kwargs):
        """
        Forward pass of the model.

        Parameters
        ----------
        x: torch.Tensor
            Input Node features size NxF
        edge_index: torch.Tensor
            Edge index of the graph size 2xE
        edge_type: torch.Tensor
            Edge type of the graph size E
        beat_nodes: torch.Tensor
            Beat nodes size B
        measure_nodes: torch.Tensor
            Measure nodes size M
        beat_index: torch.Tensor
            Beat index size 2xE_b
        measure_index: torch.Tensor
            Measure index size 2xE_m
        rel_edge: torch.Tensor (optional)
            Relative edge features E
        beat_lengths: torch.Tensor (optional)


        Returns
        -------
        h: torch.Tensor
            Output node features size NxH
        """
        # Initialization
        if self.use_metrical:
            h_beat = scatter_add(self.emb_beats(x)[beat_edges[0]], beat_edges[1], dim=0, out=torch.zeros(beat_nodes.size(0), self.num_hidden, dtype=x.dtype).to(x.device))
            h_measure = scatter_add(self.emb_measures(x)[measure_edges[0]], measure_edges[1], dim=0, out=torch.zeros(measure_nodes.size(0), self.num_hidden, dtype=x.dtype).to(x.device))
        h = x
        hs = []
        for i in range(len(self.convs) - 1):
            if i != 0 and self.use_metrical:
                beat_conv, h_beat = self.beat_convs[i-1](h_beat, h, beat_edges, beat_lengths)
                measure_conv, h_measure = self.measure_convs[i-1](h_measure, h, measure_edges, measure_lengths)
                h = self.project_metrical[i-1](torch.cat([h, beat_conv, measure_conv], dim=-1))
                h = F.normalize(F.relu(h), p=2, dim=-1)
            # First loop only updates notes
            if i == 0 and self.use_reledge:
                if self.is_heterogeneous:
                    h = self.convs[i](h, edge_index, edge_type, edge_features=rel_edge)
                else:
                    h = self.convs[i](h, edge_index, edge_features=rel_edge)
            else:
                h = self.convs[i](h, edge_index, edge_type) if self.is_heterogeneous else self.convs[i](h, edge_index)
            h = F.normalize(h, p=2, dim=-1)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            hs.append(h)
        if self.use_knowledge:
            h = self.jk(hs)
        if self.use_metrical:
            beat_conv, h_beat = self.beat_convs[-1](h_beat, h, beat_edges, beat_lengths)
            measure_conv, h_measure = self.measure_convs[-1](h_measure, h, measure_edges, measure_lengths)
            h = self.project_metrical[-1](torch.cat([h, beat_conv, measure_conv], dim=-1))
            h = F.normalize(F.relu(h), p=2, dim=-1)
        h = self.convs[-1](h, edge_index, edge_type, edge_features=rel_edge) if self.is_heterogeneous else self.convs[-1](h, edge_index, edge_features=rel_edge)
        return h


class HeteroConv(nn.Module):
    """
    Convert a Graph Convolutional module to a hetero GraphConv module.

    Parameters
    ----------
    module: torch.nn.Module
        Module to convert

    Returns
    -------
    module: torch.nn.Module
        Converted module
    """

    def __init__(self, in_features, out_features, etypes, in_edge_features=None, module:nn.Module=SageConv, bias=True, reduction='mean', activation=None):
        super(HeteroConv, self).__init__()
        self.out_features = out_features
        self.etypes = etypes
        self.activation = activation if activation is not None else nn.Identity()
        if reduction == 'mean':
            self.reduction = lambda x: x.mean(dim=0)
        elif reduction == 'sum':
            self.reduction = lambda x: x.sum(dim=0)
        elif reduction == 'max':
            self.reduction = lambda x: x.max(dim=0)
        elif reduction == 'min':
            self.reduction = lambda x: x.min(dim=0)
        elif reduction == 'concat':
            self.reduction = lambda x: torch.cat(x, dim=0)
        elif reduction == 'lstm':
            self.reduction = HeteroAttention(out_features, len(etypes.keys()))
        else:
            raise NotImplementedError

        conv_dict = dict()
        for etype in etypes.keys():
            conv_dict[etype] = module(in_features, out_features, bias=bias, in_edge_features=in_edge_features)
        self.conv = nn.ModuleDict(conv_dict)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.conv.values():
            conv.reset_parameters()

    def forward(self, x, edge_index, edge_type, edge_features=None):
        out = torch.zeros((len(self.conv.keys()), x.shape[0], self.out_features))
        for idx, (ekey, evalue) in enumerate(self.etypes.items()):
            mask = edge_type == evalue
            out[idx] = self.conv[ekey](x, edge_index[:, mask], edge_features[mask, :] if edge_features is not None else None)
            out[idx] = self.activation(out[idx])
        return self.reduction(out).to(x.device)



class HeteroMusGConv(nn.Module):
    """
    Convert a Graph Convolutional module to a hetero GraphConv module.

    Parameters
    ----------
    module: torch.nn.Module
        Module to convert

    Returns
    -------
    module: torch.nn.Module
        Converted module
    """

    def __init__(self, in_features, out_features, metadata, in_edge_features=0, bias=True, reduction='mean', return_edge_emb=False):
        super(HeteroMusGConv, self).__init__()
        self.out_features = out_features
        self.return_edge_emb = return_edge_emb
        self.etypes = metadata[1]
        if reduction == 'mean':
            self.reduction = lambda x: x.mean(dim=0)
        elif reduction == 'sum':
            self.reduction = lambda x: x.sum(dim=0)
        elif reduction == 'max':
            self.reduction = lambda x: x.max(dim=0)
        elif reduction == 'min':
            self.reduction = lambda x: x.min(dim=0)
        elif reduction == 'concat':
            self.reduction = lambda x: torch.cat(x, dim=0)
        else:
            raise NotImplementedError

        conv_dict = dict()
        for etype in self.etypes:
            etype_str = "_".join(etype)
            conv_dict[etype_str] = MusGConv(in_features, out_features, bias=bias, in_edge_channels=in_edge_features, return_edge_emb=return_edge_emb)
        self.conv = nn.ModuleDict(conv_dict)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.conv.values():
            conv.reset_parameters()

    def forward(self, x, edge_index_dict, edge_feature_dict=None):
        x = x["note"] if isinstance(x, dict) else x
        edge_feature_dict = {key: None for key in self.etypes} if edge_feature_dict is None else edge_feature_dict
        out = torch.zeros((len(self.conv.keys()), x.shape[0], self.out_features), device=x.device)
        for idx, ekey in enumerate(self.etypes):
            etype_str = "_".join(ekey)
            if self.return_edge_emb:
                out[idx], edge_feature_dict[ekey] = self.conv[etype_str](x, edge_index_dict[ekey], edge_feature_dict[ekey])
            else:
                out[idx] = self.conv[etype_str](x, edge_index_dict[ekey], edge_feature_dict[ekey])
        if self.return_edge_emb:
            return {"note": self.reduction(out)}, edge_feature_dict
        return {"note": self.reduction(out)}