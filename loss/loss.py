from typing import Any
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from utils.utils import accuracy


class FocalLoss(nn.Module):
    """
    Derived from https://github.com/yatengLG/Focal-Loss-Pytorch/blob/master/Focal_Loss.py
    """

    def __init__(self, alpha=None, gamma=2, num_classes=3, size_average="mean"):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha, list):
            assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中第一类为背景类
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """

        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1, preds.size(-1))
        alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1)  # log_softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(alpha, loss.t())
        if self.size_average == "mean":
            loss = loss.mean()
        elif self.size_average == "sum":
            loss = loss.sum()
        return loss


class IntegratedLoss(nn.Module):
    """
    This class computes the loss for Stamp Map Tracker.
    """

    def __init__(self, road_element_cls, loss_weights, label_bg_weight, pt_confidence_thres, pt_padding_value):
        super().__init__()
        self.loss_weights = loss_weights
        self.road_element_cls = road_element_cls
        self.focal_loss = FocalLoss(alpha=label_bg_weight, num_classes=len(self.road_element_cls))
        self.point_confidence_threshold = pt_confidence_thres
        self.pt_padding_value = pt_padding_value

    def loss_labels(self, src_logits, targets, indices):
        """

        :param src_logits:
        :param targets:
        :param indices: list[tuple(track_indices, gt_indices)] len(list) = batch_size
        :return:
        """
        # get the index of the matched src and gt
        idx = self._get_src_permutation_idx(indices)
        # flatten gt labels
        target_classes_o = torch.cat([t[J] for t, (_, J) in zip(targets, indices)])
        # full gt; shape like src; default value is background
        target_classes = torch.full(src_logits.shape[:2], self.road_element_cls['background'],
                                    dtype=torch.int64, device=src_logits.device)
        # only the positions with matched src have a valid gt label
        target_classes[idx] = target_classes_o

        # loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        label_loss = self.focal_loss(src_logits, target_classes)

        acc = accuracy(src_logits[idx], target_classes_o)[0]

        return label_loss, acc

    def loss_pt_coord(self, outputs, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        src_pt = outputs[idx]  # rang in [0, 1]
        target_pt = torch.cat([t[i] for t, (_, i) in zip(targets, indices)], dim=0)

        # only use points with valid coordinate to calculate loss
        mask = target_pt != self.pt_padding_value
        src_pt = src_pt[mask]
        target_pt = target_pt[mask]

        loss_pts = F.l1_loss(src_pt, target_pt, reduction='mean')

        return loss_pts

    def loss_pt_confidence(self, src_logits, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        target_pt_confidence = torch.cat([t[J] for t, (_, J) in zip(targets, indices)]).to(torch.float32)
        src_pt_confidence = src_logits[idx]

        loss_ce = F.binary_cross_entropy(src_pt_confidence, target_pt_confidence)

        return loss_ce

    def loss_curb_type(self, src_logits, targets, indices):
        idx = self._get_src_permutation_idx(indices)
        target_curb_type = torch.cat([t[J] for t, (_, J) in zip(targets, indices)])
        src_curb_type = src_logits[idx]

        loss_ce = F.cross_entropy(src_curb_type.transpose(1, 2), target_curb_type)

        return loss_ce

    @staticmethod
    def _get_src_permutation_idx(indices):
        """
        convert indices format
        :param indices: list[tuple(track_indices, gt_indices)] len(list) = batch_size
        :return:
        """
        # len(batch_idx) = total number of items in a batch
        # indicate the index of the sample in the batch
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        # len(src_idx) = total number of items in a batch
        # indicate the index of the predictions in the batch which have a matched gt
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @staticmethod
    def _get_tgt_permutation_idx(indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def forward(self, model_outputs: list, model_inputs):
        road_gt = model_inputs["road_gt"]

        loss_dict = {}

        # loop over each sample
        for i in range(len(model_outputs)):
            model_frame_output = model_outputs[i]

            matched_results = model_frame_output["matched_gt_indices"]
            # list[tuple(track_indices, gt_indices)] len(list) = batch_size
            matched_indices = [(matches[:, 0], matches[:, 1]) for matches in matched_results]

            gt_batch_num = road_gt['gt_num'][i]

            # road element class loss
            class_gt = road_gt['gt_class'][i].squeeze(-1)
            # split gt into different samples in a batch
            class_gt = torch.split(class_gt, gt_batch_num.tolist())
            class_loss, acc = self.loss_labels(model_frame_output["cls_pred"], class_gt, matched_indices)
            class_loss = self.loss_weights['class_weight'] * class_loss
            loss_dict.update({"frame_{}_class_loss".format(i): class_loss})
            # loss_dict.update({"frame_{}_class_accuracy_loss".format(i): acc})

            # road element point confidence loss
            point_confidence_gt = road_gt['gt_pt_padding_flags'][i]
            point_confidence_gt = torch.split(point_confidence_gt, gt_batch_num.tolist(), dim=0)
            point_confidence_loss = self.loss_weights['pt_confidence_weight'] * self.loss_pt_confidence(
                model_frame_output["point_confidence_pred"],
                point_confidence_gt, matched_indices)
            loss_dict.update({"frame_{}_point_confidence_loss".format(i): point_confidence_loss})

            # road element point loss
            points_gt = road_gt['gt_points'][i]
            points_gt = torch.split(points_gt, gt_batch_num.tolist(), dim=0)
            points_loss = self.loss_weights['pt_coord_weight'] * self.loss_pt_coord(
                model_frame_output["point_coord_pred"], points_gt, matched_indices)
            loss_dict.update({"frame_{}_point_coord_loss".format(i): points_loss})

            # curb type loss
            # curb_matched_indices = [(matches["curb_matched_indices"][:, 0],
            #                          matches["curb_matched_indices"][:, 1]) for matches in matched_results]
            #
            # curb_type_gt = road_gt['gt_curb_types'][i].squeeze(-1)
            # curb_type_gt = torch.split(curb_type_gt, [2, 5], dim=0)
            # curb_type_loss = self.loss_curb_type(model_frame_output["curb_type_pred"], curb_type_gt,
            #                                      curb_matched_indices)
            # loss_dict.update({"frame_{}_curb_type_loss".format(i): curb_type_loss})
            #
            # lane_matched_indices = [(matches["lane_matched_indices"][:, 0],
            #                          matches["lane_matched_indices"][:, 1]) for matches in matched_results]

        return loss_dict
