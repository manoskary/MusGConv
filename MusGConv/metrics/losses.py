import torch.nn.functional as F
import torch
from torch_scatter import scatter

class LinkPredictionLoss(torch.nn.Module):
    def __init__(self):
        super(LinkPredictionLoss, self).__init__()

    def forward(self, pos_score: torch.Tensor, neg_score: torch.Tensor):
        pos_loss = -torch.log(pos_score + 1e-15).mean()
        neg_loss = -torch.log(1 - neg_score + 1e-15).mean()
        loss = pos_loss + neg_loss
        return loss


class LinearAssignmentLoss(torch.nn.Module):
    def __init__(self):
        super(LinearAssignmentLoss, self).__init__()

    def forward(self, edge_index: torch.Tensor, score, target_edges, num_nodes):
        add_row = scatter(score, edge_index[0], dim=0, dim_size=num_nodes, reduce="sum")
        add_col = scatter(score, edge_index[1], dim=0, dim_size=num_nodes, reduce="sum")
        norm_row = torch.sqrt(scatter(torch.pow(score, 2), edge_index[0], dim=0, dim_size=num_nodes, reduce="sum"))
        norm_col = torch.sqrt(scatter(torch.pow(score, 2), edge_index[1], dim=0, dim_size=num_nodes, reduce="sum"))
        ones_row = torch.zeros((num_nodes,), device=edge_index.device)
        ones_col = torch.zeros((num_nodes,), device=edge_index.device)
        ones_row[target_edges[0]] = 1
        ones_col[target_edges[1]] = 1
        l1 = torch.sqrt(torch.sum(torch.pow(add_row - ones_row, 2))/num_nodes) + torch.sqrt(torch.sum(torch.pow(add_col - ones_col, 2))/num_nodes)
        l2 = torch.sqrt(torch.sum(torch.pow(norm_row - ones_row, 2))/num_nodes) + torch.sqrt(torch.sum(torch.pow(norm_col - ones_col, 2))/num_nodes)
        mse_1 = torch.pow(ones_row - add_row, 2).sum() + torch.pow(ones_col - add_col, 2).sum() # removed sqrt
        # mae_1 = torch.abs(ones_row - add_row).sum() + torch.abs(ones_col - add_col).sum()
        mse_2 = torch.pow(ones_row - norm_row, 2).sum() + torch.pow(ones_col - norm_col, 2).sum() # removed sqrt
        # mae_2 = torch.abs(ones_row - norm_row).sum() + torch.abs(ones_col - norm_col).sum()
        # lc = (mse_1 + mse_2) / num_nodes
        # lc = (mae_1 + mae_2) / num_nodes
        lc = torch.sqrt((mse_1 + mse_2) / (4*num_nodes))
        # lc = (l1 + l2) # / num_nodes
        return lc


class LinearAssignmentLossCE(torch.nn.Module):
    def __init__(self):
        super(LinearAssignmentLossCE, self).__init__()

    def forward(self, edge_index, score, target_edges, num_nodes):
        """Computes the loss for the linear assignment problem, using cross entropy.

        Args:
            edge_index (torch.Tensor): list of all potential edges
            edge_pred_mask_logits (torch.Tensor): unnormalized logits predicted for the potential edges. Shape (pot_edges.shape[1]).
            edge_target_mask (torch.Tensor): binary targets for the potential edges. Shape (pot_edges.shape[1]).

        Returns:
            float: the loss
        """
        loss_sum = torch.tensor(0.0, device=edge_index.device)
        # for each non-zero element, take all pot_edges that starts and ends there
        for start, dst in target_edges.T:
            # get all pot_edges that start at start and end at dst
            mask = (edge_index[0] == start) | (edge_index[1] == dst)
            edge_target_mask = (edge_index[0] == start) & (edge_index[1] == dst)
            if torch.any(edge_target_mask):
                # get the logits for the restricted pot_edges
                pred_logits = score[mask]
                # get the ground truth for the restricted pot_edges
                target_logits = edge_target_mask[mask].long()
                loss_sum += F.cross_entropy(pred_logits.unsqueeze(0), target_logits.unsqueeze(0).argmax(-1))/pred_logits.shape[0]
        loss_sum /= target_edges.shape[1]
        return loss_sum