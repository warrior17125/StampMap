"""
STAMP MAP model and criterion classes.
"""
import logging
import os.path as osp
from typing import Any
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from models.matcher import build_matcher
from models.transformer import build_transformer
from models.module import build_lifemanager, build_generator
from loss.loss import IntegratedLoss


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    

class Model(nn.Module):
    """
    Stamp Map Tracker Model.
    """

    def __init__(self, 
                 track_queries_generator,
                 transformer,              
                 clip_matcher,
                 track_query_life_manager,
                 road_pt_pad_num,
                 train_batch_size,
                 infer_batch_size,
                 train_mode,
    ) -> None:
        super().__init__()

        # model modules
        self._track_queries_generator = track_queries_generator
        self._pt_data_encoder = MLP(2 * road_pt_pad_num, 256, 256, 2)
        self._class_data_encoder = MLP(1, 256, 256, 1)
        self._transformer = transformer

        # task heads
        self._cls_predictor = MLP(256, 256, 4, 1)
        self._point_confidence_predictor = MLP(256, 256, road_pt_pad_num, 1)
        self._point_coord_predictor = MLP(256, 256, 2 * road_pt_pad_num, 1)
        self._curb_type_predictor = None
        self._curb_sub_type_predictor = None
        self._lane_type_predictor = None
        self._lane_color_predictor = None

        # matchers & managers
        self._clip_matcher = clip_matcher
        self._query_life_manager = track_query_life_manager

        # infer
        self._prev_sub_clip_id = None

        # track instance
        self._track_instance = None
        self.road_pt_pad_num = road_pt_pad_num
        self.train_mode = train_mode
        self.batch_size = train_batch_size,
        
    @staticmethod
    def adapt_inputs(pts, classes, paddings):
        """
        aggregate pts & attrs to road data
        :param classes:
        :param pts: [2, 15, 50, 350, 2]
        :param paddings: [2, 15, 50]
        :return:
        """
        class_data = torch.flatten(classes, start_dim=1, end_dim=2).unsqueeze(-1)
        # road_data = torch.concat([pts, only_class_attrs], dim=4)  # [b_s, h_f, ele_pad, pt_pad, (cls_dim+attr_dim)]
        # flatten history frames x road elements
        pt_data = torch.flatten(pts, start_dim=1, end_dim=2)  # [b_s, h_f*ele_pad, pt_pad, (cls_dim+attr_dim)]
        # flatten points x (coordinates, attrs)
        pt_data = torch.flatten(pt_data, start_dim=2, end_dim=3)  # [b_s, h_f*ele_pad, pt_pad*(cls_dim+attr_dim)]

        key_padding_mask = torch.flatten(paddings, start_dim=1, end_dim=2)  # [b_s, h_f*ele_pad]
        return pt_data, class_data, key_padding_mask

    def common_forward(self, pts, classes, paddings, delta_ts, track_instance):
        """
        model forward for one batch sample
        :param classes:
        :param pts:
        :param paddings:
        :param delta_ts:
        :param track_instance: track queries
        :return:
        """

        # generate track queries
        track_instance = self._track_queries_generator(track_instance)

        # add attrs & delta_ts to pts
        # [b_s, h_f*ele_pad, pt_pad*(cls_dim+attr_dim)] [b_s, h_f*ele_pad]
        road_data, class_data, key_padding_mask = self.adapt_inputs(pts, classes, paddings)

        # encoder road data(two MLP: pt_pad*(cls_dim+attr_dim) ——> d_model)
        pt_features = self._pt_data_encoder(road_data)  # [b_s, h_f*ele_pad, d_model]
        class_features = self._class_data_encoder(class_data)
        road_features = pt_features + class_features

        # transformer
        # fusion history frames; update track queries
        track_queries, memory = self._transformer(
            src=road_features,  # [b_s, h_f*ele_pad, d_model] : # [b_s, c, d_model]
            mask=key_padding_mask,  # [b_s, h_f*ele_pad] : [b_s, c]
            pos_embed=delta_ts,  # [b_s, h_f]
            query_embed=track_instance.query_embed,  # [b_s, track_num, d_model]
        )  # [b_s, track_num, d_model] [b_s, c, d_model]

        # task heads
        # class & point confidence & point coordinates
        cls_pred = self._cls_predictor(track_queries)  # [b_s, track_num, cls_dim]
        point_confidence_pred = self._point_confidence_predictor(track_queries).sigmoid()  # [b_s, track_num, pt_pad]
        point_coord_pred = self._point_coord_predictor(track_queries).sigmoid()
        point_coord_pred = point_coord_pred.view(self.batch_size[0], -1, self.road_pt_pad_num, 2)  # [b_s, track_num, pt_pad*2]

        # # curb attributes
        # curb_type_pred = self._curb_type_predictor(track_queries)  # [b_s, track_num, pt_pad*4] ("4" is curb types)
        # # [b_s, track_num, pt_pad, 4]
        # curb_type_pred = curb_type_pred.view(self._batch_size, -1, self.num_point, self.num_curb_type)
        # # [b_s, track_num, pt_pad*11] ("11" is curb subtypes)
        # curb_sub_type_pred = self._curb_sub_type_predictor(track_queries)
        # # [b_s, track_num, pt_pad, 11]
        # curb_sub_type_pred = curb_sub_type_pred.view(self._batch_size, -1, self.num_point, self.num_curb_subtype)
        #
        # # lane attributes
        # lane_type_pred = self._lane_type_predictor(track_queries)  # [b_s, track_num, pt_pad*13] ("13" is lane types)
        # # [b_s, track_num, pt_pad, 13]
        # lane_type_pred = lane_type_pred.view(self._batch_size, -1, self.num_point, self.num_lane_type)
        # lane_color_pred = self._lane_color_predictor(track_queries)  # [b_s, track_num, pt_pad*11] ("11" is lane color)
        # # [b_s, track_num, pt_pad, 11]
        # lane_color_pred = lane_color_pred.view(self._batch_size, -1, self.num_point, self.num_lane_color)

        track_instance.query_embed = track_queries  # [b_s, track_num, d_model]
        track_instance.track_score = cls_pred.clone().detach().softmax(dim=-1).max(dim=-1).values
        track_instance.cls_pred = cls_pred.clone().detach()
        track_instance.point_confidence_pred = point_confidence_pred.clone().detach()
        track_instance.point_coord_pred = point_coord_pred.clone().detach()

        if self.train_mode:
            # match track queries with ground truth in sub clip
            track_instance, batch_matched_indices = self._clip_matcher(track_instance)

        # track query life manager
        track_instance = self._query_life_manager(track_instance)

        if self.train_mode:
            pred_outputs = {
                "cls_pred": cls_pred,
                "point_confidence_pred": point_confidence_pred,
                "point_coord_pred": point_coord_pred,
                # "curb_type_pred": curb_type_pred,
                # "curb_sub_type_pred": curb_sub_type_pred,
                # "lane_type_pred": lane_type_pred,
                # "lane_color_pred": lane_color_pred,
                "matched_gt_indices": batch_matched_indices,
            }
        else:
            pred_outputs = {
                "cls_pred": track_instance.cls_pred,
                "point_confidence_pred": track_instance.point_confidence_pred,
                "point_coord_pred": track_instance.point_coord_pred,
                "track_score": track_instance.track_score,
                "element_id": track_instance.matched_gt_ids,
            }

        track_instance.remove("cls_pred")
        track_instance.remove("point_confidence_pred")
        track_instance.remove("point_coord_pred")

        return track_instance, pred_outputs

    def forward(self, inputs: Dict[str, Any], ):
        """
        mian forward function for stamp map tracker model
        :param inputs:
        :return:
        """
        # inputs to model(sample five frames)  # [batch_size, history_frames, element_pad_num, pt_pad_num, dim]
        road_pts = inputs["road_data_points"]  # [b_s, h_f, ele_pad, pa_pad, cls_dim]
        # road_attrs = inputs["road_data_attrs"]  # [b_s, h_f, ele_pad, pa_pad, attr_dim]
        road_class = inputs["road_data_class"]
        road_padding_flags = inputs["road_data_padding_element_flags"]  # [b_s, h_f, ele_pad]
        delta_ts = inputs["history_delta_ts"]  # [b_s, h_f]

        # initialize clip matcher using the batch data
        self._clip_matcher.initialize_for_single_clip(inputs["road_gt"])  # initiate track_instances

        device = "cuda"
        track_instance = None
        model_outputs = []
        # loop each training sample in the sub clip
        for frame_idx, (pts, classes, paddings, ts) in enumerate(zip(road_pts, road_class, road_padding_flags, delta_ts)):
            track_instance, outputs = self.common_forward(pts.to(device), classes.to(device), paddings.to(device), ts.to(device), track_instance)
            model_outputs.append(outputs)

        return model_outputs

    def infer_forward(self, **inputs):
        """
        Main inference forward function for stamp map tracker model
        :param inputs:
        :return:
        """
        # inputs to model(sample five frames)  # [batch_size, history_frames, element_pad_num, pt_pad_num, dim]
        road_pts = inputs["road_data_points"]  # [b_s, h_f, ele_pad, pa_pad, cls_dim]
        # road_attrs = inputs["road_data_attrs"]  # [b_s, h_f, ele_pad, pa_pad, attr_dim]
        road_class = inputs["road_data_class"]
        road_padding_flags = inputs["road_data_padding_element_flags"]  # [b_s, h_f, ele_pad]
        delta_ts = inputs["history_delta_ts"]  # [b_s, h_f]

        if len(road_pts) != 1 or len(road_class) != 1 or len(road_padding_flags) != 1 or len(delta_ts) != 1:
            raise ValueError("Inference only supports batch size 1")

        # set track instance
        sub_clip_id = inputs["sub_clip_id"]
        cur_ts = inputs["sub_clip_ts"][0][0]
        if sub_clip_id != self._prev_sub_clip_id:
            self._prev_sub_clip_id = sub_clip_id
            self._track_instance = None

        model_outputs = []
        # loop each training sample in the sub clip
        for frame_idx, (pts, classes, paddings, ts) in enumerate(zip(road_pts, road_class, road_padding_flags, delta_ts)):
            self._track_instance, outputs = self.common_forward(pts, classes, paddings, ts, self._track_instance)
            model_outputs.append(outputs)

        return model_outputs[0]


def build(args):
    device = torch.device(args.device)

    track_queries_generator = build_generator(args)

    transformer = build_transformer(args)

    clip_matcher = build_matcher(args)

    track_query_life_manager = build_lifemanager(args)

    model = Model(
        track_queries_generator=track_queries_generator,
        transformer=transformer,
        clip_matcher=clip_matcher,
        track_query_life_manager=track_query_life_manager,
        road_pt_pad_num =args.road_pt_pad_num,
        train_batch_size = args.train_batch_size,
        infer_batch_size = args.infer_batch_size,
        train_mode=args.train_mode
    )

    criterion = IntegratedLoss(
        road_element_cls=args.road_element_cls, 
        loss_weights=args.loss_weights, 
        label_bg_weight=args.label_loss_weight_background, 
        pt_confidence_thres=args.point_confidence_mask_threshold, 
        pt_padding_value=args.pad_pt_value_gt,
    )

    criterion.to(device)

    return model, criterion