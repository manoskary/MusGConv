from musgconv.models.core import *
from pytorch_lightning import LightningModule
from musgconv.models.core.hgnn import MetricalGNN
from musgconv.utils import add_reverse_edges_from_edge_index
from torchmetrics import Accuracy, F1Score
from musgconv.utils import METADATA
from torch_geometric.nn import to_hetero
from musgconv.models.core.utils import HeteroMusGConvEncoder, SageEncoder, GATEncoder, ResGatedConvEncoder


class ComposerClassificationModel(nn.Module):
    def __init__(self, input_features, hidden_features, output_features, num_layers, activation=F.relu,
                 dropout=0.5, use_reledge=False, metrical=False, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.return_edge_emb = kwargs.get("return_edge_emb", False)
        self.activation = activation
        self.input_features = input_features
        self.hidden_features = hidden_features
        self.output_features = output_features
        pitch_embeddding = kwargs.get("pitch_embedding", 0)
        pitch_embeddding = 0 if pitch_embeddding is None else pitch_embeddding
        self.in_edge_features = 6 + pitch_embeddding
        block = kwargs.get("conv_block", "SageConv")
        if block == "ResConv":
            print("Using ResGatedGraphConv")
            enc = ResGatedConvEncoder(input_features, hidden_features, n_layers=num_layers, dropout=dropout, activation=activation)
            self.encoder = to_hetero(enc, metadata=METADATA, aggr="mean")
        elif block == "SageConv" or block == "Sage" or block is None:
            print("Using SageConv")
            enc = SageEncoder(input_features, hidden_features, n_layers=num_layers, dropout=dropout, activation=activation)
            self.encoder = to_hetero(enc, metadata=METADATA, aggr="mean")
        elif block == "GAT" or block == "GATConv":
            print("Using GATConv")
            enc = GATEncoder(input_features, hidden_features, n_layers=num_layers, dropout=dropout, activation=activation)
            self.encoder = to_hetero(enc, metadata=METADATA, aggr="mean")
        elif block == "RelEdgeConv" or block == "MusGConv":
            print("Using MusGConv")
            self.encoder = HeteroMusGConvEncoder(input_features, hidden_features, METADATA, n_layers=num_layers, dropout=dropout, activation=activation, in_edge_features=self.in_edge_features, return_edge_emb=self.return_edge_emb)
        else:
            raise ValueError("Block type not supported")
        kwargs["conv_block"] = block
        self.clf = nn.Linear(hidden_features, output_features)

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.clf.reset_parameters()

    def forward(self, x_dict, edge_dict, edge_feature_dict, **kwargs):
        h = self.encoder(x_dict, edge_dict, edge_feature_dict)["note"]
        lengths = kwargs["lengths"].tolist() if isinstance(kwargs["lengths"], torch.Tensor) else kwargs["lengths"]
        sg = torch.split(h, lengths)
        sg = [torch.mean(s, dim=0) for s in sg]
        sg = torch.stack(sg)
        return self.clf(sg)


class ComposerClassificationModelLightning(LightningModule):
    def __init__(self, input_features, hidden_features, output_features, num_layers, activation=F.relu, dropout=0.5, lr=0.001, weight_decay=5e-4, use_jk=False, reg_loss_type="la", use_reledge=False, metrical=False, **kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation
        self.input_features = input_features
        self.hidden_features = hidden_features
        self.output_features = output_features
        self.module = ComposerClassificationModel(input_features, hidden_features, output_features, num_layers,
                                                  activation=activation, dropout=dropout,
                                                  use_reledge=use_reledge, metrical=metrical, **kwargs)
        pitch_embedding = kwargs.get("pitch_embedding", None)
        self.etypes = {"onset": 0, "consecutive": 1, "during": 2, "rests": 3, "consecutive_rev": 4, "during_rev": 5,
                       "rests_rev": 6}
        self.pitch_embedding = torch.nn.Embedding(12, 16) if pitch_embedding is not None else pitch_embedding
        self.lr = lr
        self.weight_decay = weight_decay
        self.use_jk = use_jk
        self.reg_loss_type = reg_loss_type
        self.use_signed_features = kwargs.get("use_signed_features", False)
        self.use_reledge = use_reledge
        self.train_acc = Accuracy(task="multiclass", num_classes=output_features)
        self.val_acc = Accuracy(task="multiclass", num_classes=output_features)
        self.test_acc = Accuracy(task="multiclass", num_classes=output_features)
        self.train_f1 = F1Score(task="multiclass", num_classes=output_features, average="macro")
        self.val_f1 = F1Score(task="multiclass", num_classes=output_features, average="macro")
        self.test_f1 = F1Score(task="multiclass", num_classes=output_features, average="macro")
        self.train_loss = torch.nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def training_step(self, *args, **kwargs):
        y_hat, y = self.common_step(*args, **kwargs)
        loss = self.train_loss(y_hat, y)
        acc = self.train_acc(y_hat, y)
        self.log("train_loss", loss.item(), prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_acc", acc.item(), prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, *args, **kwargs):
        y_hat, y = self.common_step(*args, **kwargs)
        loss = F.cross_entropy(y_hat, y)
        acc = self.val_acc(y_hat, y)
        fscore = self.val_f1(y_hat, y)
        self.log("val_loss", loss.item(), batch_size=1, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc.item(), batch_size=1, prog_bar=True, on_epoch=True)
        self.log("val_f1", fscore.item(), batch_size=1, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, *args, **kwargs):
        y_hat, y = self.common_step(*args, **kwargs)
        loss = F.cross_entropy(y_hat, y)
        acc = self.test_acc(y_hat, y)
        fscore = self.test_f1(y_hat, y)
        self.log("test_loss", loss.item(), batch_size=1)
        self.log("test_acc", acc.item(), batch_size=1)
        self.log("test_f1", fscore.item(), batch_size=1)
        return loss

    def common_step(self, batch, batch_idx):
        """ batch is dict with keys the variable names"""
        if batch["y"].shape[0] > 1:
            y = batch["y"]
        else:
            optional_repeat = batch["x"].shape[0] // batch["lengths"] + 1 if batch["x"].shape[0] % batch["lengths"] != 0 else batch["x"].shape[0] // batch["lengths"]
            y = batch["y"].repeat(optional_repeat)
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
        edge_feature_dict = {et: edge_features[edge_types == self.etypes[et[1]]] for et in METADATA[1]} if edge_features is not None else None
        y_hat = self.module(x_dict, edge_index_dict, edge_feature_dict, lengths=batch["lengths"])
        return y_hat, y

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)


