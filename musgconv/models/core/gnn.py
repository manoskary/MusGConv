import torch.nn as nn
import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn import MessagePassing


class SageConvLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(SageConvLayer, self).__init__()
        self.neigh_linear = nn.Linear(in_features, in_features, bias=bias)
        self.linear = nn.Linear(in_features * 2, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.linear.weight, gain=gain)
        nn.init.xavier_uniform_(self.neigh_linear.weight, gain=gain)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0.)
        if self.neigh_linear.bias is not None:
            nn.init.constant_(self.neigh_linear.bias, 0.)

    def forward(self, features, adj, neigh_feats=None):
        if neigh_feats is None:
            neigh_feats = features
        h = self.neigh_linear(neigh_feats)
        if not isinstance(adj, torch.sparse.FloatTensor):
            if len(adj.shape) == 3:
                h = torch.bmm(adj, h) / (adj.sum(dim=1).reshape((adj.shape[0], adj.shape[1], -1)) + 1)
            else:
                h = torch.mm(adj, h) / (adj.sum(dim=1).reshape(adj.shape[0], -1) + 1)
        else:
            h = torch.mm(adj, h) / (adj.to_dense().sum(dim=1).reshape(adj.shape[0], -1) + 1)
        z = self.linear(torch.cat([features, h], dim=-1))
        return z


class SageConvScatter(nn.Module):
    def __init__(self, in_features, out_features, bias=True, in_edge_features=None):
        super(SageConvScatter, self).__init__()
        self.neigh_linear = nn.Linear(in_features, in_features, bias=bias)
        self.linear = nn.Linear(in_features*2, out_features, bias=bias)
        self.in_edge_features = in_edge_features
        if in_edge_features is not None:
            self.edge_linear = nn.Linear(in_edge_features, in_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.linear.weight, gain=gain)
        nn.init.xavier_uniform_(self.neigh_linear.weight, gain=gain)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0.)
        if self.neigh_linear.bias is not None:
            nn.init.constant_(self.neigh_linear.bias, 0.)
        if self.in_edge_features is not None:
            nn.init.xavier_uniform_(self.edge_linear.weight, gain=gain)
            if self.edge_linear.bias is not None:
                nn.init.constant_(self.edge_linear.bias, 0.)

    def forward(self, features, edge_index, edge_features=None, neigh_feats=None):
        if neigh_feats is None:
            neigh_feats = features
        h = self.neigh_linear(neigh_feats)
        he = h[edge_index[1]]
        if self.in_edge_features is not None and edge_features is not None:
            edge_features = self.edge_linear(edge_features)
            he = he + edge_features
        s = scatter(he, edge_index[0], 0, out=features.clone(), reduce='mean')
        z = self.linear(torch.cat([features, s], dim=-1))
        return z


class MusGConv(MessagePassing):
    def __init__(self, in_channels, out_channels, in_edge_channels=0, bias=True, return_edge_emb=False, **kwargs):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.return_edge_emb = return_edge_emb
        self.aggregation = kwargs.get("aggregation", "cat")
        self.in_edge_channels = in_edge_channels if in_edge_channels > 0 else in_channels
        self.lin = nn.Linear(in_channels, out_channels)
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.in_edge_channels, out_channels),
            nn.ReLU(),
            nn.BatchNorm1d(out_channels),
            nn.Linear(out_channels, out_channels),
        )
        self.proj = nn.Linear(3 * out_channels, out_channels) if self.aggregation == "cat" else nn.Linear(2 * out_channels, out_channels)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.lin.weight, gain=gain)
        nn.init.xavier_uniform_(self.proj.weight, gain=gain)
        nn.init.xavier_uniform_(self.edge_mlp[0].weight, gain=gain)
        nn.init.xavier_uniform_(self.edge_mlp[3].weight, gain=gain)
        if self.bias is not None:
            self.bias.data.fill_(0)

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is None:
            edge_attr = torch.abs(x[edge_index[0]] - x[edge_index[1]])
        x = self.lin(x)
        edge_attr = self.edge_mlp(edge_attr)
        h = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        h = self.proj(torch.cat((x, h), dim=-1))
        if self.bias is not None:
            h = h + self.bias
        if self.return_edge_emb:
            return h, edge_attr
        return h

    def message(self, x_j, edge_attr):
        if self.aggregation == "cat":
            return torch.cat((x_j, edge_attr), dim=-1)
        elif self.aggregation == "add":
            return x_j + edge_attr
        elif self.aggregation == "mul":
            return x_j * edge_attr
        else:
            raise ValueError("Aggregation type not supported")


class RelEdgeConv(nn.Module):
    def __init__(self, in_node_features, out_features, bias=True, in_edge_features=None):
        super(RelEdgeConv, self).__init__()
        self.neigh_linear = nn.Linear(in_node_features, in_node_features, bias=bias)
        self.edge_linear = nn.Linear((in_node_features*2 if in_edge_features is None else in_node_features+in_edge_features), in_node_features, bias=bias)
        self.linear = nn.Linear(in_node_features*2, out_features, bias=bias)
        self.in_edge_features = in_edge_features
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.linear.weight, gain=gain)
        nn.init.xavier_uniform_(self.neigh_linear.weight, gain=gain)
        nn.init.xavier_uniform_(self.edge_linear.weight, gain=gain)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0.)
        if self.neigh_linear.bias is not None:
            nn.init.constant_(self.neigh_linear.bias, 0.)
        if self.edge_linear.bias is not None:
            nn.init.constant_(self.edge_linear.bias, 0.)

    def forward(self, features, edge_index, edge_features=None):
        h = self.neigh_linear(features)
        if self.in_edge_features is None or edge_features is None:
            edge_features = torch.abs(h[edge_index[0]] - h[edge_index[1]])
        new_h = self.edge_linear(torch.cat((h[edge_index[0]], edge_features), dim=-1))
        s = scatter(new_h, edge_index[1], 0, out=h.clone(), reduce="sum")
        z = self.linear(torch.cat([features, s], dim=-1))
        return z


class RelEdgeMLConv(nn.Module):
    """
    Relative edge conv inspired by Residual Gated Graph ConvNets with edge features.
    """
    def __init__(self, in_node_features, out_features, bias=True, in_edge_features=None):
        super(RelEdgeMLConv, self).__init__()
        if in_edge_features is None:
            in_edge_features = in_node_features
        self.W1 = nn.Linear(2*in_node_features+in_edge_features, out_features, bias=bias)
        self.W2 = nn.Linear(out_features, in_node_features, bias=bias)
        self.W3 = nn.Linear(2*in_node_features, out_features, bias=bias)
        self.W4 = nn.Linear(out_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W1.weight, gain=gain)
        nn.init.xavier_uniform_(self.W2.weight, gain=gain)
        nn.init.xavier_uniform_(self.W3.weight, gain=gain)
        if self.W1.bias is not None:
            nn.init.constant_(self.W1.bias, 0.)
        if self.W2.bias is not None:
            nn.init.constant_(self.W2.bias, 0.)
        if self.W3.bias is not None:
            nn.init.constant_(self.W3.bias, 0.)
        if self.W4.bias is not None:
            nn.init.constant_(self.W4.bias, 0.)

    def forward(self, features, edge_index, edge_features=None):
        h = features
        if edge_features is None:
            edge_features = torch.abs(features[edge_index[0]] - features[edge_index[1]])
        h_neigh = torch.cat((features[edge_index[0]], edge_features, features[edge_index[1]]), dim=-1)
        h_neigh = F.relu(self.W1(h_neigh))
        h_neigh = self.W2(h_neigh)
        s = scatter(h_neigh, edge_index[0], 0, out=h.clone(), reduce='sum')
        z = F.relu(self.W3(torch.cat([features, s], dim=-1)))
        z = self.W4(z)
        return z


class GATConvLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads=3, bias=True, dropout=0.3, negative_slope=0.2, in_edge_features=None):
        super(GATConvLayer, self).__init__()
        self.num_heads = num_heads
        self.in_features = in_features
        self.out_features = out_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.el = nn.Linear(in_features, in_features * num_heads, bias=bias)
        self.er = nn.Linear(in_features, in_features * num_heads, bias=bias)
        self.attnl = nn.Parameter(torch.FloatTensor(1, num_heads, in_features))
        self.attnr = nn.Parameter(torch.FloatTensor(1, num_heads, in_features))
        if in_edge_features is not None:
            self.attne = nn.Parameter(torch.FloatTensor(1, num_heads, in_features))
            self.fc_fij = nn.Linear(in_edge_features, in_features * num_heads, bias=bias)
        self.in_edge_feats = in_edge_features
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.softmax = nn.Softmax(dim=1)
        self.attndrop = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear.weight, gain=gain)
        nn.init.xavier_normal_(self.el.weight, gain=gain)
        nn.init.xavier_normal_(self.er.weight, gain=gain)
        nn.init.xavier_normal_(self.attnl, gain=gain)
        nn.init.xavier_normal_(self.attnr, gain=gain)
        if self.linear.bias is not None:
            nn.init.constant_(self.linear.bias, 0.)
        if self.el.bias is not None:
            nn.init.constant_(self.el.bias, 0.)
        if self.er.bias is not None:
            nn.init.constant_(self.er.bias, 0.)
        if self.in_edge_feats is not None:
            nn.init.xavier_normal_(self.fc_fij.weight, gain=gain)
            if self.fc_fij.bias is not None:
                nn.init.constant_(self.fc_fij.bias, 0.)

    def forward(self, features, edge_index, edge_features=None):
        prefix_shape = features.shape[:-1]
        fc_src = self.el(features).view(*prefix_shape, self.num_heads, self.in_features)
        fc_dst = self.er(features).view(*prefix_shape, self.num_heads, self.in_features)
        el = (fc_src[edge_index[0]] * self.attnl).sum(dim=-1).unsqueeze(-1)
        er = (fc_dst[edge_index[1]] * self.attnr).sum(dim=-1).unsqueeze(-1)
        if edge_features is not None and self.in_edge_feats is not None:
            edge_shape = edge_features.shape[:-1]
            fc_eij = self.fc_fij(edge_features).view(*edge_shape, self.num_heads, self.in_features)
            ee = (fc_eij * self.attne).sum(dim=-1).unsqueeze(-1)
            e = self.leaky_relu(el + er + ee)
        else:
            e = self.leaky_relu(el + er)
        # Not Quite the same as the Softmax in the paper.
        a = self.softmax(self.attndrop(e)).mean(dim=1)
        h = self.linear(features)
        out = scatter(a * h[edge_index[1]], edge_index[0], 0, out=h.clone(), reduce='add')
        return out


class ResGatedGraphConv(nn.Module):
    def __init__(self, in_features, out_features, bias=True, in_edge_features=None):
        super(ResGatedGraphConv, self).__init__()
        self.W1 = nn.Linear(in_features, out_features, bias=bias)
        self.W2 = nn.Linear(in_features, out_features, bias=bias)
        self.W3 = nn.Linear(in_features, out_features, bias=bias)
        self.W4 = nn.Linear(in_features, out_features, bias=bias)
        self.res = nn.Linear(in_features, out_features, bias=bias)
        self.in_edge_features = in_edge_features
        if in_edge_features is not None:
            self.W5 = nn.Linear(in_edge_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.W1.weight, gain=gain)
        nn.init.xavier_uniform_(self.W2.weight, gain=gain)
        nn.init.xavier_uniform_(self.W3.weight, gain=gain)
        nn.init.xavier_uniform_(self.W4.weight, gain=gain)
        nn.init.xavier_uniform_(self.res.weight, gain=gain)
        if self.W1.bias is not None:
            nn.init.constant_(self.W1.bias, 0.)
        if self.W2.bias is not None:
            nn.init.constant_(self.W2.bias, 0.)
        if self.W3.bias is not None:
            nn.init.constant_(self.W3.bias, 0.)
        if self.W4.bias is not None:
            nn.init.constant_(self.W4.bias, 0.)
        if self.in_edge_features is not None:
            nn.init.xavier_uniform_(self.W5.weight, gain=gain)
            if self.W5.bias is not None:
                nn.init.constant_(self.W5.bias, 0.)

    def forward(self, features, edge_index, edge_features=None, neigh_feats=None):
        if neigh_feats is None:
            neigh_feats = features
        h1 = self.W1(features)
        h2 = self.W2(neigh_feats)
        h3 = self.W3(features)[edge_index[0]]
        h4 = self.W4(features)[edge_index[1]]
        if edge_features is not None or self.in_edge_features is not None:
            h5 = self.W5(edge_features)
            z1 = torch.sigmoid(h3 + h4 + h5)
        else:
            z1 = torch.sigmoid(h3 + h4)
        h = z1 * h2[edge_index[0]]
        z = scatter(h, edge_index[1], 0, out=h1.clone(), reduce='mean')
        z = z + self.res(features)
        return z


class GCN(nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats, n_layers, activation=F.relu, dropout=0.5, jk=False):
        super(GCN, self).__init__()
        self.n_hidden = n_hidden
        self.layers = nn.ModuleList()
        self.normalize = F.normalize
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.layers.append(SageConvScatter(in_feats, n_hidden))
        for i in range(n_layers - 1):
            self.layers.append(SageConvScatter(n_hidden, n_hidden))
        if jk:
            self.use_knowledge = True
            self.jk = JumpingKnowledge(n_hidden=n_hidden, n_layers=n_layers)
        else:
            self.use_knowledge = False
        self.layers.append(SageConvScatter(n_hidden, out_feats))

    def forward(self, x, edge_index):
        h = x
        hs = []
        for conv in self.layers[:-1]:
            h = conv(h, edge_index)
            h = self.activation(h)
            h = self.normalize(h)
            h = self.dropout(h)
            hs.append(h)
        if self.use_knowledge:
            h = self.jk(hs)
        h = self.layers[-1](h, edge_index)
        return h


class OnsetEmbedding(nn.Module):
    def __init__(self, in_feats, out_feats, bias=True, add_self_loops=True):
        super(OnsetEmbedding, self).__init__()
        self.W = nn.Linear(in_feats, out_feats, bias=bias)
        self.add_self_loops = add_self_loops

    def forward(self, x, edge_index):
        # add self-loops
        if self.add_self_loops:
            self_loops = torch.arange(0, x.size(0), dtype=torch.long, device=x.device)
            self_loops = self_loops.unsqueeze(0).repeat(2, 1)
            edge_index = torch.cat([edge_index, self_loops], dim=1)
        src = x[edge_index[0]]
        dst = x[edge_index[1]]
        out = torch.abs(src - dst)
        out = scatter(out, edge_index[0], 0, out=x.clone(), reduce='mean')
        out = self.W(out)
        return out


def compare_all_elements(u, v, max_val, data_split=1):
    """
    Description.....

    Parameters
    ----------
        u:         first array to be compared (1D torch.tensor of ints)
        v:         second array to be compared (1D torch.tensor of ints)
        max_val:         the largest element in either tensorA or tensorB (real number)
        data_split:      the number of subsets to split the mask up into (int)
    Returns
    -------
        compared_inds_a:  indices of tensorA that match elements in tensorB (1D torch.tensor of ints, type torch.long)
        compared_inds_b:  indices of tensorB that match elements in tensorA (1D torch.tensor of ints, type torch.long)
    """
    compared_inds_a, compared_inds_b, inc = torch.tensor([]).to(u.device), torch.tensor([]).to(u.device), int(
        max_val // data_split) + 1
    for iii in range(data_split):
        inds_a, inds_b = (iii * inc <= u) * (u < (iii + 1) * inc), (iii * inc <= v) * (
                v < (iii + 1) * inc)
        tile_a, tile_b = u[inds_a], v[inds_b]
        tile_a, tile_b = tile_a.unsqueeze(0).repeat(tile_b.size(0), 1), torch.transpose(tile_b.unsqueeze(0), 0, 1).repeat(1,
                                                                                                                     tile_a.size(
                                                                                                                         0))
        nz_inds = torch.nonzero(tile_a == tile_b, as_tuple=False)
        nz_inds_a, nz_inds_b = nz_inds[:, 1], nz_inds[:, 0]
        compared_inds_a, compared_inds_b = torch.cat((compared_inds_a, inds_a.nonzero()[nz_inds_a]), 0), torch.cat(
            (compared_inds_b, inds_b.nonzero()[nz_inds_b]), 0)
    return compared_inds_a.squeeze().long(), compared_inds_b.squeeze().long()


class JumpingKnowledge(nn.Module):
    """
    Combines information per GNN layer with a LSTM,
    provided that all hidden representation are on the same dimension.
    """
    def __init__(self, n_hidden, n_layers):
        super(JumpingKnowledge, self).__init__()
        self.lstm = nn.LSTM(n_hidden, (n_layers*n_hidden)//2, bidirectional=True, batch_first=True)
        self.att = nn.Linear(2 * ((n_layers*n_hidden)//2), 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()
        nn.init.xavier_uniform_(self.att.weight, gain=nn.init.calculate_gain("relu"))

    def forward(self, xs):
        x = torch.stack(xs, dim=1)  # [num_nodes, num_layers, num_channels]
        alpha, _ = self.lstm(x)
        alpha = self.att(alpha).squeeze(-1)  # [num_nodes, num_layers]
        alpha = torch.softmax(alpha, dim=-1)
        return (x * alpha.unsqueeze(-1)).sum(dim=1)


class JumpingKnowledge3D(nn.Module):
    """
    Combines information per GNN layer with a LSTM,
    provided that all hidden representation are on the same dimension.
    """
    def __init__(self, n_hidden, n_layers):
        super(JumpingKnowledge3D, self).__init__()
        self.lstm = nn.LSTM(n_hidden, (n_layers*n_hidden)//2, bidirectional=True, batch_first=True)
        self.att = nn.Linear(2 * ((n_layers*n_hidden)//2), 1)
        self.n_hidden = n_hidden
        self.reset_parameters()

    def reset_parameters(self):
        self.lstm.reset_parameters()
        nn.init.xavier_uniform_(self.att.weight, gain=nn.init.calculate_gain("relu"))

    def forward(self, xs):
        x = torch.stack(xs, dim=2)  # [batch_size, num_nodes, num_layers, num_channels]
        h = torch.reshape(x, (x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))
        alpha, _ = self.lstm(h)
        alpha = self.att(alpha).squeeze(-1)  # [num_nodes, num_layers]
        alpha = torch.softmax(alpha, dim=-1)
        return (h * alpha.unsqueeze(-1)).sum(dim=1).view(x.shape[0], x.shape[1], self.n_hidden)


class GGRU(nn.Module):
    """
    A GRU inspired implementation of a GCN cell.

    h(t-1) in this case is the neighbors of node t.
    """
    def __init__(self, in_features, out_features, bias=False):
        super(GGRU, self).__init__()
        self.wr = SageConvLayer(in_features, in_features)
        self.wz = SageConvLayer(in_features, out_features)
        self.w_ni = nn.Linear(in_features*2, out_features)
        self.w_nh = nn.Linear(in_features, in_features)
        self.proj = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.w_ni.weight, gain=gain)
        nn.init.xavier_uniform_(self.w_nh.weight, gain=gain)
        nn.init.xavier_uniform_(self.proj.weight, gain=gain)

    def forward(self, x, adj):
        h = x
        r = F.sigmoid(self.wr(h, adj))
        z = F.sigmoid(self.wz(h, adj))
        h = torch.bmm(adj, self.w_nh(h)) / (adj.sum(dim=1).reshape((adj.shape[0], adj.shape[1], -1)) + 1)
        n = self.w_ni(torch.cat([x, r*h], dim=-1))
        n = F.tanh(n)
        neigh = torch.bmm(adj, x) / (adj.sum(dim=1).reshape((adj.shape[0], adj.shape[1], -1)) + 1)
        out = (1 - z)*n + z*self.proj(neigh)
        return out


class GPSLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads, activation, dropout=0.2, bias=True):
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
        activation: nn.Module
            Activation function
        dropout: float
            Dropout rate
        bias: bool
            Whether to use bias
        """
        super(GPSLayer, self).__init__()
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
        self.local = ResGatedGraphConv(in_features, out_features, bias=bias)

    def forward(self, x, edge_index):
        h_init = x
        # Local embeddings
        local_out = self.local(x, edge_index)
        local_out = self.activation(local_out)
        local_out = self.normalize_local(local_out)
        local_out = self.dropout_local(local_out)
        local_out = local_out + h_init

        # Global embeddings
        attn_out, _ = self.attn(x, x, x)
        attn_out = self.activation(attn_out)
        attn_out = self.normalize_attn(attn_out)
        attn_out = self.dropout_attn(attn_out)
        attn_out = attn_out + h_init

        # Combine
        out = local_out + attn_out
        h = self.ff1(out)
        h = self.activation(h)
        h = self.dropout_ff(h)
        h = self.ff2(h)
        h - self.dropout_ff(h)
        out = F.normalize(out + h)
        return out


class MetricalConvLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation=None, dropout=0.2, bias=True):
        super().__init__()
        self.input_dim = in_dim
        self.output_dim = out_dim
        self.activation = nn.Identity() if activation is None else activation
        self.dropout = nn.Dropout(dropout)
        self.normalize = nn.BatchNorm1d(out_dim)
        self.neigh = nn.Linear(in_dim, in_dim, bias=bias)
        self.conv_out = nn.Linear(3 * in_dim, out_dim, bias=bias)
        self.seq = SageConvScatter(in_dim, in_dim, bias=bias)

    def reset_parameters(self):
        self.neigh.reset_parameters()
        self.conv_out.reset_parameters()
        self.seq.reset_parameters()

    def forward(self, x_metrical, x, edge_index, lengths=None):
        if lengths is None:
            seq_index = torch.vstack((torch.arange(0, x_metrical.shape[0] - 1), torch.arange(1, x_metrical.shape[0])))
            # add inverse
            seq_index = torch.cat((seq_index, torch.vstack((seq_index[1], seq_index[0]))), dim=1).long()
        else:
            seq_index = []
            for i in range(len(lengths) - 1):
                seq_index.append(torch.vstack((torch.arange(lengths[i], lengths[i + 1] - 1), torch.arange(lengths[i] + 1, lengths[i + 1]))))
            seq_index = torch.cat(seq_index, dim=1)
            seq_index = torch.cat((seq_index, torch.vstack((seq_index[1], seq_index[0]))), dim=1).long()
        h_neigh = self.neigh(x)
        h_scatter = scatter(h_neigh[edge_index[0]], edge_index[1], dim=0, out=torch.zeros(x_metrical.size(0), self.input_dim, dtype=x.dtype).to(x.device))
        h_seq = self.seq(x_metrical, seq_index.to(x_metrical.device))
        h = torch.cat([h_scatter, x_metrical, h_seq], dim=1)
        h = self.conv_out(h)
        h = self.activation(h)
        h = self.normalize(h)
        h = self.dropout(h)
        out = scatter(h[edge_index[0]], edge_index[1], dim=0, out=torch.zeros(x.size(0), self.output_dim, dtype=h.dtype).to(x.device))
        return out, h
