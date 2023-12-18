from .lightning_base import VocSepLightningModule
from .VoicePred import LinkPredictionModel, HeteroLinkPredictionModel, MetricalLinkPredictionModel
from torch.nn import functional as F
from musgconv.models.core import UNet
import torch
import torch_geometric as pyg
from pytorch_lightning import LightningModule
from musgconv.utils.pianoroll import pr_to_voice_pred
from musgconv.metrics.slow_eval import AverageVoiceConsistency, MonophonicVoiceF1
from musgconv.utils import voice_from_edges, METADATA
from .VoicePredGeom import PGLinkPredictionModel
from musgconv.utils import add_reverse_edges, add_reverse_edges_from_edge_index
from torchmetrics import F1Score
from scipy.sparse.csgraph import connected_components


class VoiceLinkPredictionModel(VocSepLightningModule):
    def __init__(
        self,
        in_feats,
        n_hidden,
        n_layers,
        activation=F.relu,
        dropout=0.5,
        lr=0.001,
        weight_decay=5e-4,
        linear_assignment=True,
        model="ResConv",
        jk=True,
        reg_loss_weight="auto"
    ):
        super(VoiceLinkPredictionModel, self).__init__(
            in_feats,
            n_hidden,
            n_layers,
            activation,
            dropout,
            lr,
            weight_decay,
            LinkPredictionModel,
            linear_assignment=linear_assignment,
            model_name=model,
            jk=jk,
            reg_loss_weight=reg_loss_weight
        )

    def training_step(self, batch, batch_idx):
        batch_inputs, edges, batch_labels, edge_types, pot_edges, truth_edges, na, name = batch
        edges, edge_types = add_reverse_edges_from_edge_index(edges, edge_types)
        pos_edges = pot_edges[:, batch_labels.bool()]
        neg_labels = torch.where(~batch_labels.bool())[0]
        neg_edges = pot_edges[
            :, neg_labels[torch.randperm(len(neg_labels))][: pos_edges.shape[1]]
        ]
        h = self.module.embed(batch_inputs, edges)
        pos_pitch_score = self.pitch_score(pos_edges, na[:, 0])
        pos_onset_score = self.onset_score(pos_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        neg_pitch_score = self.pitch_score(neg_edges, na[:, 0])
        neg_onset_score = self.onset_score(neg_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        pitch_score = self.pitch_score(pot_edges, na[:, 0])
        onset_score = self.onset_score(pot_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        pos_out = self.module.predict(h, pos_edges, pos_pitch_score, pos_onset_score)
        neg_out = self.module.predict(h, neg_edges, neg_pitch_score, neg_onset_score)
        reg_loss = self.reg_loss(
            pot_edges, self.module.predict(h, pot_edges, pitch_score, onset_score), pos_edges, len(batch_inputs))
        batch_pred = torch.cat((pos_out, neg_out), dim=0)
        loss = self.train_loss(pos_out, neg_out)
        batch_pred = torch.cat((1 - batch_pred, batch_pred), dim=1).squeeze()
        targets = (
            torch.cat(
                (torch.ones(pos_out.shape[0]), torch.zeros(neg_out.shape[0])), dim=0
            )
            .long()
            .to(self.device)
        )
        self.log("train_regloss", reg_loss.item(), on_step=True, on_epoch=True, prog_bar=False, batch_size=1)
        self.log("regloss_weight", self.reg_loss_weight, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        self.log("regloss_weighted", self.reg_loss_weight*reg_loss.item(), on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        self.train_metric_logging_step(loss, batch_pred, targets)
        loss = loss + self.reg_loss_weight * reg_loss
        self.log("train_joinloss", loss.item(), on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_inputs, edges, batch_labels, edge_types, pot_edges, truth_edges, na, name = batch
        edges, edge_types = add_reverse_edges_from_edge_index(edges, edge_types)
        pitch_score = self.pitch_score(pot_edges, na[:, 0])
        onset_score = self.onset_score(pot_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        batch_pred = self.module(pot_edges, batch_inputs, edges, pitch_score, onset_score)
        self.val_metric_logging_step(
            batch_pred, pot_edges, truth_edges, len(batch_inputs)
        )

    def test_step(self, batch, batch_idx):
        batch_inputs, edges, batch_labels, edge_types, pot_edges, truth_edges, na, name = batch
        edges, edge_types = add_reverse_edges_from_edge_index(edges, edge_types)
        pitch_score = self.pitch_score(pot_edges, na[:, 0])
        onset_score = self.onset_score(pot_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        batch_pred = self.module(pot_edges, batch_inputs, edges, pitch_score, onset_score)
        self.test_metric_logging_step(
            batch_pred, pot_edges, truth_edges, len(batch_inputs)
        )

    def compute_linkpred_loss(self, pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
        )
        w_coef = pos_score.shape[0] / neg_score.shape[0]
        weight = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.ones(neg_score.shape[0]) * w_coef]
        )
        return F.binary_cross_entropy(scores.squeeze(), labels, weight=weight)



class HeteroVoiceLinkPredictionModel(VocSepLightningModule):
    def __init__(
        self,
        in_feats,
        n_hidden,
        n_layers,
        activation=F.relu,
        dropout=0.5,
        lr=0.001,
        weight_decay=5e-4,
        linear_assignment=True,
        model="ResConv",
        jk=True,
        reg_loss_weight="auto",
        reg_loss_type="la",
        tau=0.5
    ):
        super(HeteroVoiceLinkPredictionModel, self).__init__(
            in_feats,
            n_hidden,
            n_layers,
            activation,
            dropout,
            lr,
            weight_decay,
            HeteroLinkPredictionModel,
            linear_assignment=linear_assignment,
            model_name=model,
            jk=jk,
            reg_loss_weight=reg_loss_weight,
            reg_loss_type=reg_loss_type,
            tau=tau
        )

    def training_step(self, batch, batch_idx):
        batch_inputs, edges, batch_labels, edge_types, pot_edges, truth_edges, na, name = batch
        edges, edge_types = add_reverse_edges_from_edge_index(edges, edge_types)
        pos_edges = pot_edges[:, batch_labels.bool()]
        neg_labels = torch.where(~batch_labels.bool())[0]
        neg_edges = pot_edges[
            :, neg_labels[torch.randperm(len(neg_labels))][: pos_edges.shape[1]]
        ]
        h = self.module.embed(batch_inputs, edges, edge_types)
        pos_pitch_score = self.pitch_score(pos_edges, na[:, 0])
        pos_onset_score = self.onset_score(pos_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        neg_pitch_score = self.pitch_score(neg_edges, na[:, 0])
        neg_onset_score = self.onset_score(neg_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        pitch_score = self.pitch_score(pot_edges, na[:, 0])
        onset_score = self.onset_score(pot_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        pos_out = self.module.predict(h, pos_edges, pos_pitch_score, pos_onset_score)
        neg_out = self.module.predict(h, neg_edges, neg_pitch_score, neg_onset_score)
        reg_loss = self.reg_loss(
            pot_edges, self.module.predict(h, pot_edges, pitch_score, onset_score), pos_edges, len(batch_inputs))
        batch_pred = torch.cat((pos_out, neg_out), dim=0)
        loss = self.train_loss(pos_out, neg_out)
        batch_pred = torch.cat((1 - batch_pred, batch_pred), dim=1).squeeze()
        targets = (
            torch.cat(
                (torch.ones(pos_out.shape[0]), torch.zeros(neg_out.shape[0])), dim=0
            )
            .long()
            .to(self.device)
        )
        self.log("train_regloss", reg_loss.item(), on_step=True, on_epoch=True, prog_bar=False, batch_size=1)
        self.log("regloss_weight", self.reg_loss_weight, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        self.log("regloss_weighted", self.reg_loss_weight*reg_loss.item(), on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        self.train_metric_logging_step(loss, batch_pred, targets)
        loss = loss + self.reg_loss_weight * reg_loss
        self.log("train_joinloss", loss.item(), on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        return loss

    def validation_step(self, batch, batch_idx):
        batch_inputs, edges, batch_labels, edge_types, pot_edges, truth_edges, na, name = batch
        edges, edge_types = add_reverse_edges_from_edge_index(edges, edge_types)
        pitch_score = self.pitch_score(pot_edges, na[:, 0])
        onset_score = self.onset_score(pot_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        batch_pred = self.module(pot_edges, batch_inputs, edges, edge_types, pitch_score, onset_score)
        self.val_metric_logging_step(
            batch_pred, pot_edges, truth_edges, len(batch_inputs)
        )

    def test_step(self, batch, batch_idx):
        batch_inputs, edges, batch_labels, edge_types, pot_edges, truth_edges, na, name = batch
        edges, edge_types = add_reverse_edges_from_edge_index(edges, edge_types)
        pitch_score = self.pitch_score(pot_edges, na[:, 0])
        onset_score = self.onset_score(pot_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        batch_pred = self.module(pot_edges, batch_inputs, edges, edge_types, pitch_score, onset_score)
        self.test_metric_logging_step(
            batch_pred, pot_edges, truth_edges, len(batch_inputs)
        )

    def predict_step(self, batch, batch_idx):
        batch_inputs, edges, batch_labels, edge_types, pot_edges, truth_edges, na, name = batch
        edges, edge_types = add_reverse_edges_from_edge_index(edges, edge_types)
        pitch_score = self.pitch_score(pot_edges, na[:, 0])
        onset_score = self.onset_score(pot_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        batch_pred = self.module(pot_edges, batch_inputs, edges, edge_types, pitch_score, onset_score)
        adj_pred, fscore = self.predict_metric_step(
            batch_pred, pot_edges, truth_edges, len(batch_inputs)
        )
        print(f"Piece {name} F-score: {fscore}")
        nov_pred, voices_pred = connected_components(csgraph=adj_pred, directed=False, return_labels=True)
        adj_target = pyg.utils.to_dense_adj(truth_edges, max_num_nodes=len(batch_inputs)).squeeze().long().cpu()
        nov_target, voices_target = connected_components(csgraph=adj_target, directed=False, return_labels=True)
        return (
            name,
            voices_pred,
            voices_target,
            nov_pred,
            nov_target,
            na[:, 1],
            na[:, 2],
            na[:, 0],
        )

    def compute_linkpred_loss(self, pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
        )
        w_coef = pos_score.shape[0] / neg_score.shape[0]
        weight = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.ones(neg_score.shape[0]) * w_coef]
        )
        return F.binary_cross_entropy(scores.squeeze(), labels, weight=weight)



class MetricalVoiceLinkPredictionModel(VocSepLightningModule):
    def __init__(
        self,
        in_feats,
        n_hidden,
        n_layers,
        activation=F.relu,
        dropout=0.5,
        lr=0.001,
        weight_decay=5e-4,
        **kwargs
    ):
        super(MetricalVoiceLinkPredictionModel, self).__init__(
            in_feats,
            n_hidden,
            n_layers,
            activation,
            dropout,
            lr,
            weight_decay,
            MetricalLinkPredictionModel,
            **kwargs
        )
        self.etypes = {"onset":0, "consecutive":1, "during":2, "rests":3, "consecutive_rev":4, "during_rev":5, "rests_rev":6}
        self.use_signed_features = kwargs.get("use_signed_features", False)
        pitch_embedding = kwargs.get("pitch_embedding", None)
        self.pitch_embedding = torch.nn.Embedding(12, 16) if pitch_embedding is not None else pitch_embedding

    def training_step(self, batch, batch_idx):
        edges, edge_types = add_reverse_edges_from_edge_index(batch["edge_index"], batch["edge_type"])
        pos_edges = batch["potential_edges"][:, batch["y"].bool()]
        neg_labels = torch.where(~batch["y"].bool())[0]
        neg_edges = batch["potential_edges"][
            :, neg_labels[torch.randperm(len(neg_labels))][: pos_edges.shape[1]]
        ]
        na = batch["note_array"]
        if self.use_reledge:
            edge_features = na[:, :5][edges[1]] - na[:, :5][edges[0]]
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
        h = self.module.encoder(x_dict, edge_index_dict, edge_feature_dict)["note"]
        pos_pitch_score = self.pitch_score(pos_edges, na[:, 0])
        pos_onset_score = self.onset_score(pos_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        neg_pitch_score = self.pitch_score(neg_edges, na[:, 0])
        neg_onset_score = self.onset_score(neg_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        pitch_score = self.pitch_score(batch["potential_edges"], na[:, 0])
        onset_score = self.onset_score(batch["potential_edges"], na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        pos_out = self.module.predict(h, pos_edges, pos_pitch_score, pos_onset_score)
        neg_out = self.module.predict(h, neg_edges, neg_pitch_score, neg_onset_score)
        reg_loss = self.reg_loss(
            batch["potential_edges"], self.module.predict(h, batch["potential_edges"],
                                                          pitch_score, onset_score), pos_edges, len(batch["x"]))
        batch_pred = torch.cat((pos_out, neg_out), dim=0)
        loss = self.train_loss(pos_out, neg_out)
        batch_pred = torch.cat((1 - batch_pred, batch_pred), dim=1).squeeze()
        targets = (
            torch.cat(
                (torch.ones(pos_out.shape[0]), torch.zeros(neg_out.shape[0])), dim=0
            )
            .long()
            .to(self.device)
        )
        self.log("train_regloss", reg_loss.item(), on_step=True, on_epoch=True, prog_bar=False, batch_size=1)
        self.log("regloss_weight", self.reg_loss_weight, on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        self.log("regloss_weighted", self.reg_loss_weight*reg_loss.item(), on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        self.train_metric_logging_step(loss, batch_pred, targets)
        loss = loss + self.reg_loss_weight * reg_loss
        self.log("train_joinloss", loss.item(), on_step=False, on_epoch=True, prog_bar=False, batch_size=1)
        return loss

    def validation_step(self, batch, batch_idx):
        edges, edge_types = add_reverse_edges_from_edge_index(batch["edge_index"], batch["edge_type"])
        na = batch["note_array"]
        if self.use_reledge:
            edge_features = na[:, :5][edges[1]] - na[:, :5][edges[0]]
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
        pitch_score = self.pitch_score(batch["potential_edges"], na[:, 0])
        onset_score = self.onset_score(batch["potential_edges"], na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        batch_pred = self.module(x_dict, edge_index_dict, edge_feature_dict, batch["potential_edges"], pitch_score, onset_score,
                                 )
        self.val_metric_logging_step(
            batch_pred, batch["potential_edges"], batch["truth_edges"], len(batch["x"])
        )

    def test_step(self, batch, batch_idx):
        edges, edge_types = add_reverse_edges_from_edge_index(batch["edge_index"], batch["edge_type"])
        na = batch["note_array"]
        if self.use_reledge:
            edge_features = na[:, :5][edges[1]] - na[:, :5][edges[0]]
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
        pitch_score = self.pitch_score(batch["potential_edges"], na[:, 0])
        onset_score = self.onset_score(batch["potential_edges"], na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        batch_pred = self.module(x_dict, edge_index_dict, edge_feature_dict, batch["potential_edges"], pitch_score,
                                 onset_score,
                                 )
        self.test_metric_logging_step(
            batch_pred, batch["potential_edges"], batch["truth_edges"], len(batch["x"])
        )

    def predict_step(self, batch, batch_idx):
        batch_inputs, edges, batch_labels, edge_types, pot_edges, truth_edges, na, name, beat_nodes, beat_index, measure_nodes, measure_index = batch
        edges, edge_types = add_reverse_edges_from_edge_index(edges, edge_types)
        pitch_score = self.pitch_score(pot_edges, na[:, 0])
        onset_score = self.onset_score(pot_edges, na[:, 1], na[:, 2], na[:, 3], na[:, 4], na[:, 5])
        batch_pred = self.module(pot_edges, batch_inputs, edges, edge_types, pitch_score, onset_score, beat_nodes, measure_nodes, beat_index,
                                                              measure_index)
        adj_pred, fscore = self.predict_metric_step(
            batch_pred, pot_edges, truth_edges, len(batch_inputs)
        )
        print(f"Piece {name} F-score: {fscore}")
        nov_pred, voices_pred = connected_components(csgraph=adj_pred, directed=False, return_labels=True)
        adj_target = pyg.utils.to_dense_adj(truth_edges, max_num_nodes=len(batch_inputs)).squeeze().long().cpu()
        nov_target, voices_target = connected_components(csgraph=adj_target, directed=False, return_labels=True)
        return (
            name,
            voices_pred,
            voices_target,
            nov_pred,
            nov_target,
            na[:, 1],
            na[:, 2],
            na[:, 0],
        )

    def compute_linkpred_loss(self, pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
        )
        w_coef = pos_score.shape[0] / neg_score.shape[0]
        weight = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.ones(neg_score.shape[0]) * w_coef]
        )
        return F.binary_cross_entropy(scores.squeeze(), labels, weight=weight)


class VoiceLinkPredictionLightModelPG(VocSepLightningModule):
    def __init__(
        self,
        graph_metadata,
        in_feats,
        n_hidden,
        n_layers=2,
        activation=F.relu,
        dropout=0.5,
        lr=0.001,
        weight_decay=5e-4,
        linear_assignment=False,
        rev_edges=None,
        pos_weight = None,
        jk_mode = "lstm",
        conv_type = "gcn",
    ):
        super(VoiceLinkPredictionLightModelPG, self).__init__(
            in_feats, n_hidden, n_layers, activation, dropout, lr, weight_decay, None, linear_assignment
        )
        self.save_hyperparameters()
        self.module = PGLinkPredictionModel(
            graph_metadata,
            in_feats,
            n_hidden,
            n_layers,
            activation=activation,
            dropout=dropout,
            jk_mode = jk_mode,
            conv_type = conv_type,
        )
        print(f"Graph edge types: {graph_metadata}")
        self.rev_edges = rev_edges
        self.train_loss = torch.nn.BCEWithLogitsLoss(pos_weight= torch.tensor([pos_weight]))
        # self.train_loss_func = F.binary_cross_entropy_with_logits

    def training_step(self, batch, batch_idx):
        graph = batch[0]
        if self.rev_edges is not None:
            add_reverse_edges(graph, mode=self.rev_edges)
        pot_edges = graph["pot_edges"]
        edge_target_mask = graph["truth_edges_mask"]
        onsets = graph["note"].onset_div
        durations = graph["note"].duration_div
        pitches = graph["note"].pitch
        onset_beats = graph["note"].onset_beat
        duration_beats = graph["note"].duration_beat
        ts_beats = graph["note"].ts_beats
        edge_pred_mask_logits = self.module(
            graph.x_dict, graph.edge_index_dict, pot_edges, onsets, durations, pitches, onset_beats, duration_beats, ts_beats
        )
        loss = self.train_loss(edge_pred_mask_logits.float(), edge_target_mask.float())
        # get predicted class for the edges (e.g. 0 or 1)
        edge_pred__mask_normalized = torch.sigmoid(edge_pred_mask_logits)
        edge_pred_mask_bool = torch.round(edge_pred__mask_normalized).bool()
        self.train_metric_logging_step(loss, edge_pred_mask_bool, edge_target_mask)
        return loss

    def validation_step(self, batch, batch_idx):
        graph = batch[0]
        if self.rev_edges is not None:
            add_reverse_edges(graph, mode=self.rev_edges)
        pot_edges = graph["pot_edges"]
        num_notes = len(graph.x_dict["note"])
        edge_target = graph["truth_edges"]
        onsets = graph["note"].onset_div
        durations = graph["note"].duration_div
        pitches = graph["note"].pitch
        onset_beats = graph["note"].onset_beat
        duration_beats = graph["note"].duration_beat
        ts_beats = graph["note"].ts_beats
        edge_pred_mask_logits = self.module(
            graph.x_dict, graph.edge_index_dict, pot_edges, onsets, durations, pitches, onset_beats, duration_beats, ts_beats
        )
        edge_pred__mask_normalized = torch.sigmoid(edge_pred_mask_logits)
        self.val_metric_logging_step(
            edge_pred__mask_normalized, pot_edges, edge_target, num_notes, linear_assignment=self.linear_assignment
        )

    def test_step(self, batch, batch_idx):
        graph = batch[0]
        if self.rev_edges is not None:
            add_reverse_edges(graph, mode=self.rev_edges)
        pot_edges = graph["pot_edges"]
        num_notes = len(graph.x_dict["note"])
        edge_target = graph["truth_edges"]
        onsets = graph["note"].onset_div
        durations = graph["note"].duration_div
        pitches = graph["note"].pitch
        onset_beats = graph["note"].onset_beat
        duration_beats = graph["note"].duration_beat
        ts_beats = graph["note"].ts_beats
        edge_pred_mask_logits = self.module(
            graph.x_dict, graph.edge_index_dict, pot_edges, onsets, durations, pitches, onset_beats, duration_beats, ts_beats
        )
        edge_pred__mask_normalized = torch.sigmoid(edge_pred_mask_logits)
        # log without linear assignment
        self.test_metric_logging_step(
            edge_pred__mask_normalized, pot_edges, edge_target, num_notes
        )
        # log with linear assignment
        self.test_metric_logging_step(
            edge_pred__mask_normalized, pot_edges, edge_target, num_notes
        )

    def predict_step(self, batch, batch_idx):
        graph = batch[0]
        if self.rev_edges is not None:
            add_reverse_edges(graph, mode=self.rev_edges)
        pot_edges = graph["pot_edges"]
        num_notes = len(graph.x_dict["note"])
        edge_target = graph["truth_edges"]
        onsets = graph["note"].onset_div
        durations = graph["note"].duration_div
        pitches = graph["note"].pitch
        onset_beats = graph["note"].onset_beat
        duration_beats = graph["note"].duration_beat
        ts_beats = graph["note"].ts_beats
        edge_pred_mask_logits = self.module(
            graph.x_dict, graph.edge_index_dict, pot_edges, onsets, durations, pitches, onset_beats, duration_beats, ts_beats
        )
        edge_pred__mask_normalized = torch.sigmoid(edge_pred_mask_logits)
        adj_pred, fscore = self.predict_metric_step(
            edge_pred__mask_normalized, pot_edges, edge_target, num_notes
        )
        print(f"Piece {graph['name']} F-score: {fscore}")
        nov_pred, voices_pred = connected_components(csgraph=adj_pred, directed=False, return_labels=True)
        adj_target = pyg.utils.to_dense_adj(edge_target, max_num_nodes=num_notes).squeeze().long().cpu()
        nov_target, voices_target = connected_components(csgraph=adj_target, directed=False, return_labels=True)
        return ( 
            voices_pred, 
            voices_target, 
            nov_pred, 
            nov_target, 
            graph["note"].onset_div,
            graph["note"].duration_div,
            graph["note"].pitch
        )


class UnetVoiceSeparationModel(LightningModule):
    def __init__(
        self,
        n_classes,
        input_channels=1,
        lr=0.0005,
        weight_decay=5e-4,
    ):
        super(UnetVoiceSeparationModel, self).__init__()
        self.save_hyperparameters()
        self.module = UNet(input_channels, n_classes)
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
        # self.train_f1 = F1Score(num_classes=2, average="macro")
        self.val_avc = AverageVoiceConsistency(allow_permutations=False)
        self.val_monophonic_f1 = MonophonicVoiceF1(num_classes=2, average="macro")

    def training_step(self, batch, batch_idx):
        pr_dict = batch[0]
        voice_pr = pr_dict["voice_pianoroll"].squeeze().T
        input_pr = (
            torch.clip(voice_pr + 1, 0, 1)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device, dtype=torch.float64)
        )
        labels = voice_pr.to(self.device).unsqueeze(0)

        pred = self.module(input_pr)
        loss = self.train_loss(pred, labels)
        # batch_f1 = self.train_f1(batch_pred, batch_labels)
        self.log(
            "train_loss",
            loss.item(),
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            batch_size=1,
            sync_dist=True,
        )
        # self.log("train_f1", batch_f1.item(), prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pr_dict = batch[0]
        voice_pr = pr_dict["voice_pianoroll"].squeeze().T
        input_pr = (
            torch.clip(voice_pr + 1, 0, 1)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device, dtype=torch.float64)
        )
        labels = voice_pr.to(self.device).unsqueeze(0)

        pred = self.module(input_pr)
        loss = self.train_loss(pred, labels)
        self.log(
            "val_loss",
            loss.item(),
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            batch_size=1,
        )
        voice_pred = pr_to_voice_pred(
            F.log_softmax(pred.squeeze(), dim=0),
            pr_dict["notearray_onset_beat"].squeeze(),
            pr_dict["notearray_duration_beat"].squeeze(),
            pr_dict["notearray_pitch"].squeeze(),
            piano_range=True,
            time_div=12,
        )
        voice_pred = voice_pred.to(self.device)
        fscore = self.val_monophonic_f1(
            voice_pred,
            pr_dict["notearray_voice"].squeeze(),
            pr_dict["notearray_onset_beat"].squeeze(),
            pr_dict["notearray_duration_beat"].squeeze(),
        )
        self.log(
            "val_f1",
            fscore.item(),
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            batch_size=1,
            sync_dist=True,
        )
        avc = self.val_avc(voice_pred, pr_dict["notearray_voice"].squeeze())
        self.log(
            "val_avc",
            avc.item(),
            prog_bar=True,
            on_epoch=True,
            on_step=True,
            batch_size=1,
            sync_dist=True,
        )
        # add F1 computation
        return avc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return {
            "optimizer": optimizer,
        }
