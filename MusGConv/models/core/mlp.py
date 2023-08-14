import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_feats, n_hidden, out_feats=None, n_layers=1, activation=F.relu, dropout=0.5, bias=False):
        super(MLP, self).__init__()
        if out_feats is None:
            out_feats = n_hidden
        self.n_hidden = n_hidden
        self.layers = nn.ModuleList()
        self.normalize = nn.BatchNorm1d(n_hidden)
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.layers.append(nn.Linear(in_feats, n_hidden, bias=bias))
        for i in range(n_layers - 1):
            self.layers.append(nn.Linear(n_hidden, n_hidden, bias=bias))
        self.layers.append(nn.Linear(n_hidden, out_feats, bias=bias))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(len(self.layers)):
            nn.init.xavier_uniform_(self.layers[i].weight, gain=nn.init.calculate_gain('relu'))
            if self.layers[i].bias is not None:
                nn.init.constant_(self.layers[i].bias, 0.)

    def forward(self, x):
        h = x
        for layer in self.layers[:-1]:
            h = layer(h)
            h = self.activation(h)
            h = self.normalize(h)
            h = self.dropout(h)
        h = self.layers[-1](h)
        return h


