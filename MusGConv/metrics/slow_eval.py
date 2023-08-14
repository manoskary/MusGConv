from torchmetrics import Metric
from torchmetrics.classification.f_beta import F1Score
from sklearn.metrics import accuracy_score
import torch
import numpy as np
import itertools
from typing import Any, Optional

# class AverageVoiceConsistency(Metric):
#     """Implementation of the evaluation metric proposed by the paper
#     'Chew, E., & Wu, X. (2005). Separating voices in polyphonic music:
#     A contig mapping approach. In U. Wiil (Ed.), Computer music modeling and retrieval,
#     Vol. 3310 (pp. 1-20). Berlin, Heidelberg: Springer.'
#     """
#     higher_is_better = True
#     full_state_update: bool = False
#     correct: torch.Tensor
#     total: torch.Tensor

#     def __init__(self, piano_range = True, time_div = 12, allow_permutations = True):
#         super(AverageVoiceConsistency, self).__init__()
#         self.piano_range = piano_range
#         self.time_div = time_div
#         self.allow_permutations = allow_permutations
#         self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

#     def update(self, pianoroll: torch.Tensor, onset_beat: torch.Tensor, duration_beat: torch.Tensor, pitch: torch.Tensor, voice_truth: torch.Tensor):
#         voice_pred = pr_to_voice_pred(pianoroll, onset_beat, duration_beat, pitch, self.piano_range, self.time_div).to(self.device)

#         if not self.allow_permutations:
#             local_correct = torch.sum(voice_pred == voice_truth)
#         else:
#             unique_voices = torch.unique(voice_truth)
#             permutations = list(itertools.permutations(unique_voices))
#             best_voice_perm_result = torch.zeros(len(permutations), dtype = torch.int64).to(self.device)
#             # iterate over all possible permutations of the prediction
#             for i_perm, perm in enumerate(permutations):
#                 perm_pred = torch.zeros_like(voice_pred, device = self.device)
#                 for e in unique_voices:
#                     perm_pred[voice_pred==e] = perm[e]
#                 best_voice_perm_result[i_perm] = torch.sum(perm_pred == voice_truth)
#             # now pick the best mapping
#             local_correct = torch.max(best_voice_perm_result)

#         self.correct += local_correct
#         self.total += len(voice_pred)

#     def compute(self):
#         return self.correct.float() / self.total


class AverageVoiceConsistency(Metric):
    """Implementation of the evaluation metric proposed by the paper
    'Chew, E., & Wu, X. (2005). Separating voices in polyphonic music:
    A contig mapping approach. In U. Wiil (Ed.), Computer music modeling and retrieval,
    Vol. 3310 (pp. 1-20). Berlin, Heidelberg: Springer.'
    """

    higher_is_better = True
    full_state_update: bool = False
    correct: torch.Tensor
    total: torch.Tensor

    def __init__(self, allow_permutations=True):
        super(AverageVoiceConsistency, self).__init__()
        self.allow_permutations = allow_permutations
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, voice_pred: torch.Tensor, voice_truth: torch.Tensor):
        if not self.allow_permutations:
            local_correct = torch.sum(voice_pred == voice_truth)
        else:
            unique_voices = torch.unique(voice_truth)
            permutations = list(itertools.permutations(unique_voices))
            best_voice_perm_result = torch.zeros(
                len(permutations), dtype=torch.int64
            ).to(self.device)
            # iterate over all possible permutations of the prediction
            for i_perm, perm in enumerate(permutations):
                perm_pred = torch.zeros_like(voice_pred, device=self.device)
                for e in unique_voices:
                    perm_pred[voice_pred == e] = perm[e]
                best_voice_perm_result[i_perm] = torch.sum(perm_pred == voice_truth)
            # now pick the best mapping
            local_correct = torch.max(best_voice_perm_result)

        self.correct += local_correct
        self.total += len(voice_pred)

    def compute(self):
        return self.correct.float() / self.total


class MonophonicVoiceF1(F1Score):
    """Implementation of monophonic F1-score for voice separation
    """

    def __init__(self, **kwargs: Any):
        super(MonophonicVoiceF1, self).__init__(**kwargs)

    # def update(self, voice_pred: torch.Tensor, voice_truth: torch.Tensor, onset, duration):
    #     edge_matrix_pred = torch.zeros((len(voice_pred),len(voice_pred)), dtype= torch.int32).to(self.device)
    #     edge_matrix_truth = torch.zeros((len(voice_truth),len(voice_truth)), dtype = torch.int32).to(self.device)
    #     for i_start, (start_pred, start_truth) in enumerate(zip(voice_pred, voice_truth)):
    #         for i_end, (end_pred, end_truth) in enumerate(zip(voice_pred,voice_truth)):
    #             if (onset[i_start] + duration[i_start] == onset[i_end]):
    #                 if (start_pred == end_pred):
    #                     edge_matrix_pred[i_start, i_end] = 1
    #                 if (start_truth == end_truth):
    #                     edge_matrix_truth[i_start, i_end] = 1
        
    #     super(MonophonicVoiceF1, self).update(edge_matrix_pred.flatten(), edge_matrix_truth.flatten())

    def update(self, voice_pred: torch.Tensor, voice_truth: torch.Tensor, onset, duration):
        len_na = len(voice_pred)
        # build a len_na x len_na matrix where consecutive notes are True
        consecutive = onset == (onset+duration).expand(len_na,len_na).t()
        # build a len_na x len_na matrix where same voice notes are True
        same_voice_pred = voice_pred == voice_pred.expand(len_na,len_na).t()
        same_voice_truth = voice_truth == voice_truth.expand(len_na,len_na).t()
        # find consecutive notes on the same voice
        edge_matrix_pred = torch.logical_and(consecutive,same_voice_pred)
        edge_matrix_truth = torch.logical_and(consecutive, same_voice_truth)
        
        super(MonophonicVoiceF1, self).update(edge_matrix_pred.flatten(), edge_matrix_truth.flatten())