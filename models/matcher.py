import copy
import logging
from typing import Optional, List

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn, Tensor

from utils.utils import Instances
from loss.loss import FocalLoss

logger = logging.getLogger(__name__)


class HungarianMatcher(nn.Module):
    """
    Derived from DETR HungarianMatcher class.
    https://github.com/facebookresearch/detr/blob/main/models/matcher.py

    This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, match_weights, road_element_cls, label_bg_weight):
        """
        Creates the matcher

        :param match_weights: weights of matching terms
        """

        super().__init__()
        self.cost_class = match_weights['class_weight']
        self.cost_pt_coord = match_weights['pt_coord_weight']
        self.cost_pt_confidence = match_weights['pt_confidence_weight']
        assert self.cost_class != 0 or self.cost_pt_coord != 0 or self.cost_pt_confidence != 0, "all costs cant be 0"

        self.focal_loss = FocalLoss(alpha=label_bg_weight, num_classes=len(road_element_cls), size_average=None)

    def forward(self, outputs, targets):
        """
        Performs the matching
        :param outputs: This is a dict that contains at least these entries: "pred_class", "pred_pt_confidence", "pred_pt_coord"
        :param targets: unmatched gt instances
        :return: A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
                For each batch element, it holds:
                    len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        with (torch.no_grad()):
            device = outputs["pred_pt_confidence"].device
            out_class = outputs["pred_class"]
            out_pt_confidence = outputs["pred_pt_confidence"]
            out_pt_coord = outputs["pred_pt_coord"].flatten(1, 2)
            num_queries = out_pt_confidence.shape[0]

            tgt_class = targets.labels.squeeze(-1)
            tgt_pt_padding_mask = targets.points_padding_mask
            tgt_pt_coord = targets.points[:, :, 0:2].flatten(1, 2)
            num_gt = tgt_pt_padding_mask.shape[0]

            # Compute the classification cost.
            out_class_expand = out_class.unsqueeze(0).repeat(num_gt, 1, 1)
            gt_class_expand = tgt_class.unsqueeze(-1).repeat(1, num_queries)
            cost_class = self.focal_loss(out_class_expand, gt_class_expand).view(num_gt, num_queries).t()

            # Compute the point confidence cost.
            out_pt_confidence = out_pt_confidence.unsqueeze(1).expand(-1, num_gt, -1)
            tgt_pt_confidence = tgt_pt_padding_mask.unsqueeze(0).expand(num_queries, -1, -1)
            cost_pt_confidence = (tgt_pt_confidence - out_pt_confidence).abs().sum(-1)

            # Compute the point coordinate cost.
            out_pt_coords = out_pt_coord.unsqueeze(1).expand(-1, num_gt, -1)
            tgt_pt_coords = tgt_pt_coord.unsqueeze(0).expand(num_queries, -1, -1)
            cost_pt_coords_all = (out_pt_coords - tgt_pt_coords)
            # need to filter padding gt points; similar to loss calculation
            # 1.0 is a magic number; since pred points are normalized to [0, 1], and gt padding points are -1.0
            # hence det point - padding gt point > 1.0 ([0, 1] - (-1) > 1.0)
            # we use cost_pt_coords_all < 1.0 to mask all padding point positions
            # TODO: need to check the magic number 1.0
            coords_mask = cost_pt_coords_all < 1.0
            valid_pt_cnt = coords_mask.sum(-1)
            cost_pt_coords_filter = torch.where(coords_mask, cost_pt_coords_all, torch.tensor(0.0, device=device))
            cost_pt_coords_sum = cost_pt_coords_filter.abs().sum(dim=-1)
            cost_pt_coords = cost_pt_coords_sum / valid_pt_cnt

            # Final cost matrix
            C = self.cost_class * cost_class + self.cost_pt_coord * cost_pt_coords # + self.cost_pt_confidence * cost_pt_confidence
            C = C.cpu()

            indices = linear_sum_assignment(C)

            return [torch.as_tensor(indices[0], dtype=torch.int64),
                    torch.as_tensor(indices[1], dtype=torch.int64)]


class ClipMatcher(nn.Module):
    """
    clip matcher; match the track queries with the gt instances in a sub clip
    """

    def __init__(self, num_samples, batch_size, num_query, match_weights, road_element_cls, label_bg_weight):
        """

        :param num_samples: number of samples in a sub clip
        :param batch_size:
        :param num_query:
        :param match_weights:
        """

        super().__init__()
        self.gt_instances = []
        self.matcher = HungarianMatcher(match_weights, road_element_cls, label_bg_weight)
        self._current_frame_idx = 0
        self._num_samples = num_samples
        self.batch_size = batch_size
        self.num_query = num_query

    def _step(self):
        self._current_frame_idx += 1

    def initialize_for_single_clip(self, road_gt):
        """
        initialize gt instances for a single clip
        :param road_gt: gt data for the whole sub clip
        :return:
        """

        self.gt_instances = []
        self._current_frame_idx = 0
        for idx in range(self._num_samples):  # track连续五帧
            gt_instance = Instances("gt")
            gt_instance.obj_ids = road_gt["gt_ids"][idx]
            gt_instance.labels = road_gt["gt_class"][idx]
            gt_instance.points = road_gt["gt_points"][idx]
            gt_instance.points_padding_mask = road_gt["gt_pt_padding_flags"][idx]
            gt_instance.batch_num = road_gt["gt_num"][idx]
            self.gt_instances.append(gt_instance)

    def _match_new_gts(self, unmatched_predictions, unmatched_gt_instances, unmatched_track_indices,
                       unmatched_gt_indices):
        """
        match the new gt instances with the unmatched track queries; using the hungarian matcher.
        :param unmatched_predictions:
        :param unmatched_gt_instances:
        :param unmatched_track_indices:
        :param unmatched_gt_indices:
        :return:
        """
        device = unmatched_predictions["pred_class"].device

        # debug test
        # gt_coords = unmatched_gt_instances.points.clone()
        # gt_confidence = unmatched_gt_instances.points_padding_mask.clone().to(dtype=torch.float32)
        # gt_class = nn.functional.one_hot(unmatched_gt_instances.labels.clone().squeeze(-1), num_classes=4).to(dtype=torch.float32)
        # unmatched_predictions['pred_class'] = torch.cat([unmatched_predictions['pred_class'], gt_class], dim=0)
        # unmatched_predictions['pred_pt_confidence'] = torch.cat([unmatched_predictions['pred_pt_confidence'], gt_confidence], dim=0)
        # unmatched_predictions['pred_pt_coord'] = torch.cat([unmatched_predictions['pred_pt_coord'], gt_coords], dim=0)

        new_track_indices = self.matcher(unmatched_predictions, unmatched_gt_instances)

        pred_indices = new_track_indices[0]
        gt_indices = new_track_indices[1]
        # concat pred and gt.
        new_matched_indices = torch.stack([unmatched_track_indices[pred_indices], unmatched_gt_indices[gt_indices]],
                                          dim=1).to(device)
        return new_matched_indices

    @staticmethod
    def _sort_matched_indices(matched_indices):
        """
        sort matched indices by target indices.
        :param matched_indices:
        :return:
        """
        src_indices = matched_indices[:, 0]
        target_indices = matched_indices[:, 1]
        target_indices_sorted, sorted_indices = torch.sort(target_indices)
        src_indices_sorted = src_indices[sorted_indices]
        matched_indices_sorted = torch.cat([src_indices_sorted.unsqueeze(-1), target_indices_sorted.unsqueeze(-1)],
                                           dim=1)
        return matched_indices_sorted

    def _match_single_sample(self, track_instances, sample_gt_instance, sample_idx):
        """
        match gt and track queries for a single sample in a batch
        :param track_instances:
        :param sample_gt_instance: gt instance in this sample
        :param sample_idx: sample index in a batch
        :return:
        """

        device = track_instances.query_embed.device
        matched_gt_ids = track_instances.matched_gt_ids[sample_idx].clone()
        matched_gt_indices = torch.full((self.num_query,), -1, dtype=torch.long, device=device)

        # step1. inherit and update the previous matches.
        # check previous matched gt ids, update with current gt indices
        gt_obj_ids = sample_gt_instance.obj_ids
        track_matched_gt_id = matched_gt_ids.unsqueeze(-1)
        i, j = torch.where(track_matched_gt_id == gt_obj_ids)
        matched_gt_indices[i] = j

        # get overall previous matched track query to gt indices map
        full_track_indices = torch.arange(self.num_query, dtype=torch.long, device=device)
        matched_track_indices = matched_gt_indices >= 0
        # [track_indices, gt_indices] paris
        # gt instances which are previously existed but not in the current frame will not appear in prev_matched_indices
        prev_matched_indices = torch.stack(
            [full_track_indices[matched_track_indices], matched_gt_indices[matched_track_indices]],
            dim=1
        )

        # step2. get the unmatched track query indices.
        unmatched_track_indices = full_track_indices[matched_gt_ids == -1]

        # step3. select the unmatched gt instances (new tracks).
        gt_matched_indices = matched_gt_indices[matched_gt_indices != -1]
        all_gt_match_status = torch.full((len(sample_gt_instance),), True, dtype=torch.bool, device=device)
        all_gt_match_status[gt_matched_indices] = False
        unmatched_gt_indices = torch.arange(len(sample_gt_instance), device=device)[all_gt_match_status]
        if len(unmatched_gt_indices) == 0:
            return self._sort_matched_indices(prev_matched_indices)
        unmatched_gt_instances = sample_gt_instance[unmatched_gt_indices]

        # step4. match the unmatched pairs.
        unmatched_predictions = {
            "pred_class": track_instances.cls_pred[sample_idx, unmatched_track_indices],
            "pred_pt_confidence": track_instances.point_confidence_pred[sample_idx, unmatched_track_indices],
            "pred_pt_coord": track_instances.point_coord_pred[sample_idx, unmatched_track_indices],
        }
        new_matched_indices = self._match_new_gts(unmatched_predictions, unmatched_gt_instances,
                                                  unmatched_track_indices, unmatched_gt_indices)

        # step5. update obj_idxes according to the new matching result.
        matched_gt_ids[new_matched_indices[:, 0]] = sample_gt_instance.obj_ids[new_matched_indices[:, 1]].long()
        track_instances.matched_gt_ids[sample_idx] = matched_gt_ids

        # step7. merge the unmatched pairs and the matched pairs.
        matched_indices = torch.cat([new_matched_indices, prev_matched_indices], dim=0)

        # step8. sort the matched indices.
        matched_indices_sorted = self._sort_matched_indices(matched_indices)

        return matched_indices_sorted

    def train_forward(self, track_instances):
        """
        match the track queries with the gt instances for a single sample
        :param track_instances:
        :return:
        """

        with (torch.no_grad()):
            gt_instances_batch = self.gt_instances[self._current_frame_idx]
            device = track_instances.query_embed.device

            # get gt indices in each sample in a batch
            gt_batch_num = gt_instances_batch.batch_num
            batch_gt_cum_sum = torch.cat(
                (torch.zeros(1, device=device, dtype=torch.int32), torch.cumsum(gt_batch_num, dim=0))
            )

            batch_matched_indices = []
            # loop samples in a batch
            for sample_idx in range(self.batch_size):
                gt_sample_idxes = torch.arange(start=batch_gt_cum_sum[sample_idx], end=batch_gt_cum_sum[sample_idx + 1],
                                               device=device)
                sample_matched_indices = self._match_single_sample(track_instances, gt_instances_batch[gt_sample_idxes],
                                                                   sample_idx)
                batch_matched_indices.append(sample_matched_indices)

            self._step()
            return track_instances, batch_matched_indices


def build_matcher():
    return ClipMatcher()