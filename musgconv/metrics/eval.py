from torchmetrics import Metric, Accuracy
import torch
from sklearn.metrics import roc_auc_score
from torch_scatter import scatter


class VoiceSeparationAUC(Metric):
    def __init__(self):
        super(VoiceSeparationAUC, self).__init__()
        self.add_state("correct", default=torch.tensor(0, dtype=float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pos_score: torch.Tensor, neg_score: torch.Tensor):
        scores = torch.cat([pos_score, neg_score]).numpy()
        labels = torch.cat([torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]).numpy()
        assert labels.shape == scores.shape

        score = roc_auc_score(labels, scores)
        self.correct += score
        self.total += 1

    def compute(self):
        return self.correct.float() / self.total


class LinearAssignmentScore(Metric):
    def __init__(self):
        super(LinearAssignmentScore, self).__init__()
        self.add_state("correct", default=torch.tensor(0, dtype=float), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, edge_index: torch.Tensor, score, target_edges, num_nodes):
        score = (score > 0.3).float()
        add_row = scatter(score, edge_index[0], dim=0, dim_size=num_nodes, reduce="sum")
        add_col = scatter(score, edge_index[1], dim=0, dim_size=num_nodes, reduce="sum")
        ones = torch.zeros((num_nodes,), device=edge_index.device)
        ones[torch.unique(torch.cat([target_edges[0], target_edges[1]], dim=0))] = 1
        score = torch.sqrt(torch.pow(ones - add_row, 2).sum()) + torch.sqrt(torch.pow(ones - add_col, 2).sum())
        self.correct += (score / num_nodes).item()
        self.total += 1

    def compute(self):
        return self.correct.float() / self.total


class MultitaskAccuracy(Metric):
    def __init__(self, tasks, ignore_index=-1):
        super(MultitaskAccuracy, self).__init__()
        self.accs = torch.nn.ModuleDict()
        for task in tasks.keys():
            self.accs[task] = Accuracy(task="multiclass", ignore_index=ignore_index, num_classes=tasks[task])

    def update(self, pred, target):
        for task in self.accs.keys():
            self.accs[task].update(pred[task], target[task])

    def compute(self):
        return {task: self.accs[task].compute() for task in self.accs.keys()}
