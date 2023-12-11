from musgconv.models.core import *
from pytorch_lightning import LightningModule
from musgconv.models.core.hgnn import MetricalGNN, HeteroMusGConv
from musgconv.utils import add_reverse_edges_from_edge_index, METADATA
from torchmetrics import Accuracy, F1Score
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import to_hetero, SAGEConv, GATConv


class HeteroSageEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, metadata=METADATA, n_layers=2, dropout=0.5, activation=F.relu, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(to_hetero(SAGEConv(in_channels, out_channels, aggr="sum"), metadata, aggr="mean"))
        if n_layers > 1:
            for i in range(n_layers-1):
                self.layers.append(to_hetero(SAGEConv(out_channels, out_channels, aggr="sum"), metadata, aggr="mean"))
        self.dropout = dropout
        self.activation = activation

    def reset_parameters(self):
        for conv in self.layers:
            conv.reset_parameters()

    def forward(self, x_dict, edge_index_dict, edge_feature_dict, **kwargs):
        for conv in self.layers[:-1]:
            x_dict = conv(x_dict, edge_index_dict, edge_feature_dict)
            x_dict = {k: F.normalize(v, dim=-1) for k, v in x_dict.items()}
            x_dict = {k: self.activation(v) for k, v in x_dict.items()}
            x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}
        x_dict = self.layers[-1](x_dict, edge_index_dict, edge_feature_dict)
        return x_dict["note"]


class HeteroGATEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, metadata, n_layers=2, dropout=0.5, activation=F.relu, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(to_hetero(GATConv(in_channels, out_channels, aggr="sum"), metadata, aggr="mean"))
        if n_layers > 1:
            for i in range(n_layers - 1):
                self.layers.append(to_hetero(GATConv(out_channels, out_channels, aggr="sum"), metadata, aggr="mean"))
        self.dropout = dropout
        self.activation = activation

    def reset_parameters(self):
        for conv in self.layers:
            conv.reset_parameters()

    def forward(self, x_dict, edge_index_dict, edge_feature_dict, **kwargs):
        for conv in self.layers[:-1]:
            x_dict = conv(x_dict, edge_index_dict, edge_feature_dict)
            x_dict = {k: F.normalize(v, dim=-1) for k, v in x_dict.items()}
            x_dict = {k: self.activation(v) for k, v in x_dict.items()}
            x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}
        x_dict = self.layers[-1](x_dict, edge_index_dict, edge_feature_dict)
        return x_dict["note"]



class HeteroMusGConvEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, metadata, n_layers=2, dropout=0.5, activation=F.relu, **kwargs):
        super().__init__()
        self.in_edge_features = kwargs.get("in_edge_features", 0)
        self.return_edge_emb = kwargs.get("return_edge_emb", False)
        self.layers = nn.ModuleList()
        self.layers.append(HeteroMusGConv(in_channels, out_channels, metadata, in_edge_features=self.in_edge_features, return_edge_emb=self.return_edge_emb))
        if n_layers > 2:
            for i in range(n_layers - 2):
                self.layers.append(HeteroMusGConv(
                    out_channels, out_channels, metadata,
                    in_edge_features=(out_channels if self.return_edge_emb else 0),
                    return_edge_emb=self.return_edge_emb))
        self.layers.append(HeteroMusGConv(out_channels, out_channels, metadata, in_edge_features=(out_channels if self.return_edge_emb else 0), return_edge_emb=False))
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
                edge_feature_dict = {k: None for k in edge_feature_dict.keys()}
            x_dict = {k: F.normalize(v, dim=-1) for k, v in x_dict.items()}
            x_dict = {k: self.activation(v) for k, v in x_dict.items()}
            x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}
        x_dict = self.layers[-1](x_dict, edge_index_dict, edge_feature_dict)
        return x_dict["note"]


class CadenceClassificationModel(nn.Module):
    def __init__(self, input_features, hidden_features, output_features, num_layers, activation=F.relu,
                 dropout=0.5, use_reledge=False, metrical=False, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation
        self.input_features = input_features
        self.hidden_features = hidden_features
        self.output_features = output_features
        self.spelling_embedding = nn.Embedding(49, 16)
        self.pitch_embedding = nn.Embedding(128, 16)
        self.embedding = nn.Linear(input_features - 3, 32)
        pitch_embeddding = kwargs.get("pitch_embedding", 0)
        pitch_embeddding = 0 if pitch_embeddding is None else pitch_embeddding
        block = kwargs.get("conv_block", "SageConv")
        if block == "ResConv":
            conv_block = ResGatedGraphConv
        elif block == "SageConv" or block == "Sage" or block is None:
            self.encoder = HeteroSageEncoder(input_features, hidden_features, METADATA, n_layers=num_layers, dropout=dropout, activation=activation, **kwargs)
        elif block == "GAT" or block == "GATConv":
            self.encoder = HeteroGATEncoder(input_features, hidden_features, METADATA, n_layers=num_layers, dropout=dropout, activation=activation, **kwargs)
        elif block == "RelEdgeConv" or block == "MusGConv":
            self.encoder = HeteroMusGConvEncoder(input_features, hidden_features, METADATA, n_layers=num_layers, dropout=dropout, activation=activation, **kwargs)
        else:
            raise ValueError("Block type not supported")
        kwargs["conv_block"] = conv_block
        self.etypes = {"onset":0, "consecutive":1, "during":2, "rests":3, "consecutive_rev":4, "during_rev":5, "rests_rev":6}

        self.decoder_left = nn.Linear(hidden_features, hidden_features)
        self.decoder_right = nn.Linear(hidden_features, hidden_features)
        self.clf = SageConvLayer(hidden_features, output_features)
        self.smote_layer = SMOTE(dims=hidden_features, k=3)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.clf.reset_parameters()
        self.decoder_left.reset_parameters()
        self.decoder_right.reset_parameters()

    def decoder_forward(self, h):
        h = F.normalize(h, dim=-1)
        h_left = self.decoder_left(h)
        h_right = self.decoder_right(h)
        new_adj = torch.mm(h_left, h_right.T)
        new_adj = torch.sigmoid(new_adj)
        return new_adj

    def forward(self, x_dict, edge_dict, edge_feature_dict, **kwargs):
        h = self.encode(x_dict, edge_dict, edge_feature_dict)
        h_new, y_new = self.smote_layer.fit_generate(h, kwargs["y"])
        new_adj = self.decoder_forward(h_new)
        out = self.clf(h_new, F.hardshrink(new_adj, lambd=0.5))
        return out, y_new, new_adj[:h.shape[0], :h.shape[0]], h

    def encode(self, x_dict, edge_index_dict, edge_feature_dict, **kwargs):
        x = x_dict["note"]
        h_pitch = self.pitch_embedding(x[:, 0].long())
        h_spelling = self.spelling_embedding(x[:, 1].long())
        h = self.embedding(x[:, 2:-1])
        h = torch.cat([h, h_pitch, h_spelling], dim=-1)
        x_dict = {"note": h}
        h = self.encoder(x_dict, edge_index_dict, edge_feature_dict)
        return h

    def predict(self, x_dict, edge_index_dict, edge_feature_dict):
        h = self.encode(x_dict, edge_index_dict, edge_feature_dict)
        new_adj = self.decoder_forward(h)
        out = self.clf(h, new_adj)
        return out


class CadenceClassificationModelLightning(LightningModule):
    def __init__(self, input_features, n_hidden, output_features, n_layers, activation=F.relu, dropout=0.5, lr=0.001, weight_decay=5e-4, use_jk=False, use_reledge=False, metrical=False, **kwargs):
        super().__init__()
        self.num_layers = n_layers
        self.dropout = dropout
        self.activation = activation
        self.input_features = input_features
        self.hidden_features = n_hidden
        self.output_features = output_features
        self.module = CadenceClassificationModel(input_features, n_hidden, output_features, n_layers,
                                                 activation=activation, dropout=dropout,
                                                 use_reledge=use_reledge, metrical=metrical, **kwargs)
        pitch_embedding = kwargs.get("pitch_embedding", None)
        self.use_signed_features = kwargs.get("use_signed_features", False)
        self.pitch_embedding = torch.nn.Embedding(12, 16) if pitch_embedding is not None else pitch_embedding
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_jk = use_jk
        self.reg_loss_weight = kwargs.get("reg_loss_weight", 0.5)
        self.use_reledge = use_reledge
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()
        self.train_f1 = F1Score(num_classes=output_features, average="macro")
        self.val_f1 = F1Score(num_classes=output_features, average="macro")
        self.test_f1 = F1Score(num_classes=output_features, average="macro")
        self.train_loss = torch.nn.CrossEntropyLoss()
        self.train_regloss = torch.nn.BCELoss()
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx, **kwargs):
        x_dict, edge_index_dict, edge_feature_dict = self.common_step(batch, batch_idx)
        y_hat, y, new_adj, features = self.module(x_dict=x_dict, edge_index=edge_index_dict, edge_type=edge_feature_dict, y=batch["y"])
        old_adj = to_dense_adj(batch["edge_index"], max_num_nodes=batch["x"].shape[0])
        reg_loss = self.train_regloss(new_adj, old_adj.squeeze(0))
        clf_loss = self.train_loss(y_hat, y)
        feature_penalty = features.float().pow(2).mean()
        loss = clf_loss + self.reg_loss_weight*reg_loss + 0.1*feature_penalty
        acc = self.train_acc(y_hat, y)
        fscore = self.train_f1(y_hat, y)
        self.log("train_loss", loss.item(), prog_bar=True, on_epoch=True, on_step=False, batch_size=batch["x"].shape[0])
        self.log("train_clf_loss", clf_loss.item(), prog_bar=True, on_epoch=True, on_step=False, batch_size=batch["x"].shape[0])
        self.log("train_reg_loss", reg_loss.item(), prog_bar=True, on_epoch=True, on_step=False, batch_size=batch["x"].shape[0])
        self.log("train_feature_penalty", feature_penalty.item(), prog_bar=True, on_epoch=True, on_step=False, batch_size=batch["x"].shape[0])
        self.log("train_acc", acc.item(), prog_bar=True, on_epoch=True, on_step=False, batch_size=batch["x"].shape[0])
        self.log("train_f1", fscore.item(), prog_bar=True, on_epoch=True, on_step=False, batch_size=batch["x"].shape[0])
        return loss

    def validation_step(self, batch, batch_idx, **kwargs):
        x_dict, edge_index_dict, edge_feature_dict = self.common_step(
            batch, batch_idx)
        y = batch["y"]
        y_hat = self.module.predict(x_dict=x_dict, edge_index=edge_index_dict, edge_type=edge_feature_dict)

        loss = F.cross_entropy(y_hat, y)
        acc = self.val_acc(y_hat, y)
        fscore = self.val_f1(y_hat, y)
        self.log("val_loss", loss.item(), prog_bar=True, on_epoch=True, batch_size=batch["x"].shape[0])
        self.log("val_acc", acc.item(), prog_bar=True, on_epoch=True, batch_size=batch["x"].shape[0])
        self.log("val_f1", fscore.item(), prog_bar=True, on_epoch=True, batch_size=batch["x"].shape[0])

    def test_step(self, batch, batch_idx):
        x_dict, edge_index_dict, edge_feature_dict = self.common_step(
            batch, batch_idx)
        y = batch["y"]
        y_hat = self.module.predict(x_dict=x_dict, edge_index=edge_index_dict, edge_type=edge_feature_dict)
        loss = F.cross_entropy(y_hat, y)
        acc = self.test_acc(y_hat, y)
        fscore = self.test_f1(y_hat, y)
        self.log("test_loss", loss.item(), batch_size=batch["x"].shape[0])
        self.log("test_acc", acc.item(), batch_size=batch["x"].shape[0])
        self.log("test_f1", fscore.item(), batch_size=batch["x"].shape[0])

    def common_step(self, batch, batch_idx):
        """ batch is dict with keys the variable names"""
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
        edge_feature_dict = {et: edge_features[edge_types == self.etypes[et[1]]] for et in METADATA[1]}
        return x_dict, edge_index_dict, edge_feature_dict

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


