from musgconv.models.core import *
from pytorch_lightning import LightningModule
from musgconv.models.core.hgnn import MetricalGNN
from musgconv.utils import add_reverse_edges_from_edge_index
from torchmetrics import Accuracy, F1Score
from musgconv.utils import METADATA
from torch_geometric.nn import to_hetero, global_mean_pool
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
        self.in_edge_features = 5 + pitch_embeddding if use_reledge else 0
        aggregation = kwargs.get("aggregation", "cat")
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
            self.encoder = HeteroMusGConvEncoder(input_features, hidden_features, METADATA, n_layers=num_layers, dropout=dropout, activation=activation, in_edge_features=self.in_edge_features, return_edge_emb=self.return_edge_emb, aggregation=aggregation)
        else:
            raise ValueError("Block type not supported")
        kwargs["conv_block"] = block
        self.clf = nn.Sequential(
            nn.Linear(hidden_features, hidden_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features // 2, output_features))

    def forward(self, x_dict, edge_dict, edge_feature_dict, batch, **kwargs):
        h = self.encoder(x_dict, edge_dict, edge_feature_dict)["note"]
        sg = global_mean_pool(h, batch)
        return self.clf(sg)

    def forward_old(self, x_dict, edge_dict, edge_feature_dict, **kwargs):
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
        self.use_wandb = kwargs.get("use_wandb", False)
        self.module = ComposerClassificationModel(input_features, hidden_features, output_features, num_layers,
                                                  activation=activation, dropout=dropout,
                                                  use_reledge=use_reledge, metrical=metrical, **kwargs)
        pitch_embedding = kwargs.get("pitch_embedding", None)
        self.etypes = {"onset": 0, "consecutive": 1, "during": 2, "rest": 3, "consecutive_rev": 4, "during_rev": 5,
                       "rest_rev": 6}
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
        batch_size = y_hat.shape[0]
        loss = self.train_loss(y_hat, y)
        acc = self.train_acc(y_hat, y)
        self.log("train_loss", loss.item(), batch_size=batch_size, prog_bar=True, on_epoch=True, on_step=False)
        self.log("train_acc", acc.item(), batch_size=batch_size, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, *args, **kwargs):
        y_hat, y = self.common_step(*args, **kwargs)
        batch_size = y_hat.shape[0]
        loss = F.cross_entropy(y_hat, y)
        acc = self.val_acc(y_hat, y)
        fscore = self.val_f1(y_hat, y)
        self.log("val_loss", loss.item(), batch_size=batch_size, prog_bar=True, on_epoch=True)
        self.log("val_acc", acc.item(), batch_size=batch_size, prog_bar=True, on_epoch=True)
        self.log("val_f1", fscore.item(), batch_size=batch_size, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, *args, **kwargs):
        y_hat, y = self.common_step(*args, **kwargs)
        batch_size = y_hat.shape[0]
        loss = F.cross_entropy(y_hat, y)
        acc = self.test_acc(y_hat, y)
        fscore = self.test_f1(y_hat, y)
        self.log("test_loss", loss.item(), batch_size=batch_size)
        self.log("test_acc", acc.item(), batch_size=batch_size)
        self.log("test_f1", fscore.item(), batch_size=batch_size)
        # Log WANDB table
        # columns = ["ground_truth", "prediction"]
        # data = [[y[i].item(), y_hat[i].argmax().item()] for i in range(len(y))]
        # if self.use_wandb:
        #     self.logger.log_table(key="test_table",  columns=columns, data=data)
        return loss

    def common_step(self, batch, batch_idx):
        y = batch["y"]
        x_dict = batch.x_dict
        edge_index_dict = batch.edge_index_dict
        # add reverse edges
        reverse_edges = {(k[0], k[1] + "_rev", k[2]): torch.vstack((v[1], v[0])) for k, v in edge_index_dict.items() if k[1] != "onset"}
        edge_index_dict.update(reverse_edges)
        note_array = torch.vstack((batch["note"].pitch, batch["note"].onset_div, batch["note"].duration_div,
                                batch["note"].onset_beat, batch["note"].duration_beat)).t()
        edge_feature_dict = {}
        if self.use_reledge:
            for k, edges in edge_index_dict.items():
                edge_features = note_array[edges[1]] - note_array[edges[0]]
                edge_features = edge_features if self.use_signed_features else torch.abs(edge_features)
                edge_features = F.normalize(edge_features, dim=0)
                edge_feature_dict[k] = edge_features
        else:
            for k, edges in edge_index_dict.items():
                edge_feature_dict[k] = None
        if self.pitch_embedding is not None and edge_features is not None:
            for k, edges in edge_index_dict.items():
                pitch = self.pitch_embedding(torch.remainder(note_array[:, 0][edges[0]] - note_array[:, 0][edges[1]], 12).long())
                edge_feature_dict[k] = torch.cat([edge_feature_dict[k], pitch], dim=1)
        y_hat = self.module(x_dict, edge_index_dict, edge_feature_dict, batch=batch["note"].batch)
        return y_hat, y

    def common_step_old(self, batch, batch_idx):
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


