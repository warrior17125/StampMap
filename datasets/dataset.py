# -*- coding: utf-8 -*-
"""Dataset of Stamp."""
# pylint: disable=line-too-long
# flake8:noqa
import copy
import logging
import os
import pickle
from random import randint
from typing import Any
from typing import Dict
from typing import List
from typing import Union

from pathlib import Path
import numpy as np
import torch

logger = logging.getLogger(__name__)


class LaneDataset:
    """
    dataset for stamp map tracker
    """

    def __init__(
            self,
            mode: str,
            sample_history_frame_size,
            sub_clip_sample_size,
            sample_interval,
            road_element_pad_num,
            road_pt_pad_num,
            road_element_cls,
            pad_pt_value_det,
            pad_pt_value_gt,
            pad_attr_value,
            road_element_attr_dim,
            road_element_attr_def,
            bev_range,
            attrs_range,
            task_name,
            data_meta,
            transforms: Union[List[Any], Dict[str, Any], None] = None,
    ):
        """

        :param mode: Mode in _VALID_MODE.
        :param sample_history_frame_size: number of history frames for one training sample
        :param sub_clip_sample_size: number of samples in a video sub-clip, treated as final training unit
        :param sample_interval: interval to sample frames from a video sub-clip
        :param road_element_pad_num: target number of road elements to pad to (num of track queries)
        :param road_pt_pad_num: target number of points to pad to for each road element
        :param road_element_cls: class flag for road elements
        :param pad_pt_value: value to pad for points
        :param pad_attr_value: value to pad for road element attributes
        :param road_element_attr_dim: dimension of road element attributes matrix
        :param road_element_attr_def: definition for road element attributes
        :param bev_range: range of BEV map (x_min, x_max, y_min, y_max)
        :param attrs_range: range of road element attributes (class, lane type, lane color, curb type, curb subtype)
        :param task_name: task name of this training task
        :param data_meta: dataset config
        :param transforms: data augmentation settings
        """

        super().__init__(
            mode=mode,
            meta_file=None,
            transforms=transforms,
        )

        # task settings
        self.task_name = task_name
        self.data_meta = data_meta
        self.samples_total = {}
        self.video_sub_clips = []
        self.sub_clip_max_idx = {}

        # stamp map settings
        self.sample_history_frame_size = sample_history_frame_size
        self.road_element_pad_num = road_element_pad_num
        self.road_pt_pad_num = road_pt_pad_num
        self.road_element_cls = road_element_cls
        self.pad_pt_value_det = pad_pt_value_det
        self.pad_pt_value_gt = pad_pt_value_gt
        self.pad_attr_value = pad_attr_value
        self.road_element_attr_dim = road_element_attr_dim
        self.road_element_attr_def = road_element_attr_def
        self.sub_clip_sample_size = sub_clip_sample_size
        self.sample_interval = sample_interval
        self.bev_range = bev_range
        self.attrs_range = attrs_range
        self.class_normalize_value = {
            'curb': (self.road_element_cls['curb'] - self.attrs_range["class"][0]) / (
                        self.attrs_range["class"][1] - self.attrs_range["class"][0]),
            'lane': (self.road_element_cls['lane'] - self.attrs_range["class"][0]) / (
                        self.attrs_range["class"][1] - self.attrs_range["class"][0]),
            'stopline': (self.road_element_cls['stopline'] - self.attrs_range["class"][0]) / (
                        self.attrs_range["class"][1] - self.attrs_range["class"][0])
        }

        # statistics
        self.dataset_cnt = 0
        self.total_clip_cnt = 0
        self.total_drop_frame_cnt = 0
        self.total_sample_cnt = 0
        self.total_video_sub_clip_cnt = 0
        self.clip_id_set = set()

        self.use_remote_data = False
        if "use_remote_data" in self.data_meta and self.data_meta["use_remote_data"]:
            self.use_remote_data = True
        # load annotation files
        logger.info("Dataset Init begin!")
        self.prepare_all_data()
        logger.info("Dataset Init successfully!")

    def __len__(self) -> int:
        """Get length."""
        return self.total_video_sub_clip_cnt

    def _reset_dataset_profile(self):
        """Reset_dataset_profile."""
        self.dataset_cnt = 0
        self.total_clip_cnt = 0
        self.total_drop_frame_cnt = 0
        self.total_sample_cnt = 0
        self.clip_id_set = set()

    def prepare_all_data(self):
        """
        generate training data from annotation files
        :return:
        """
        # load all samples from annotation files
        # each sample is a collection of consecutive history frames
        for name, info in self.data_meta.items():
            if name == "use_remote_data":
                continue
            self.parse_one_dataset(name, info)
            self.dataset_cnt += 1
        self.total_sample_cnt = sum([len(item) for item in self.samples_total.values()])

        # generate training sub clips from samples
        # each training sub clip contains (self.sub_clip_sample_size) samples
        # self.video_sub_clips: list of (sub_clip_id, start_idx)
        for sub_clip_id, sub_clip_data in self.samples_total.items():
            sub_clip_sz = len(sub_clip_data)
            self.sub_clip_max_idx[sub_clip_id] = sub_clip_sz - 1
            for t in range(0, sub_clip_sz - self.sub_clip_sample_size + 1):
                self.video_sub_clips.append((sub_clip_id, t))
        self.total_video_sub_clip_cnt = len(self.video_sub_clips)

        logger.info(
            "Dataset Summary: "
            "dataset number {}; "
            "Load total samples {} from {} clips; "
            "Drop total frames {}; "
            "Total valid video subclip {}".format(
                self.dataset_cnt,
                self.total_sample_cnt,
                self.total_clip_cnt,
                self.total_drop_frame_cnt,
                self.total_video_sub_clip_cnt,
            )
        )

    def _parse_single_anno(self, anno_path, root_path, data_name):
        """
        parse single annotation file; generate training samples
        :param anno_path: annotation file path
        :param root_path: root path of dataset
        :param data_name: name of dataset
        :return:
        """

        # load annotation data from disk
        with open(anno_path, "rb") as fin:
            anno_data = pickle.load(fin)
            anno_data = anno_data["train_data"]
        # repeated clip check
        drop_frame_cnt = 0
        clip_del_list = []
        for clip_id in anno_data.keys():
            if clip_id in self.clip_id_set:
                logger.warning("annotation file {} has repeated clip id {}".format(anno_path, clip_id))
                drop_frame_cnt += len(anno_data[clip_id])
                clip_del_list.append(clip_id)
            else:
                self.clip_id_set.add(clip_id)
        for clip_id in clip_del_list:
            del anno_data[clip_id]
        sub_dir = anno_path.split("/")[-2]
        clip_cnt = len(anno_data)

        # loop each clip in annotation data
        for clip_id, clip_data in anno_data.items():
            clip_ts_list = list(clip_data.keys())
            total_frame_cnt = len(clip_ts_list)

            # successive time check
            # slice clip data into multiple sub clips if time interval is too large
            successive_ts_list = []
            start_idx = 0
            for idx in range(total_frame_cnt - 1):
                if clip_ts_list[idx + 1] - clip_ts_list[idx] > 0.2:
                    successive_ts_list.append(clip_ts_list[start_idx: idx + 1])
                    start_idx = idx + 1
            if not successive_ts_list:
                successive_ts_list.append(clip_ts_list)

            # generate training sample
            for idx, ts_list in enumerate(successive_ts_list):
                sub_clip_id = "{}_{}".format(clip_id, idx)
                self.samples_total[sub_clip_id] = []

                ts_list_sz = len(ts_list)
                sample_list = [
                    ts_list[i: i + self.sample_history_frame_size] for i in
                    range(0, ts_list_sz - self.sample_history_frame_size + 1)
                ]

                for sample_ts in sample_list:
                    sample_data = {"frame_path": []}
                    for ts in sample_ts:
                        if self.use_remote_data is False:
                            sample_data["frame_path"].append(os.path.join(root_path, clip_data[ts]))
                        else:
                            sample_data["frame_path"].append(clip_data[ts])

                    sample_data["clip_idx"] = idx
                    sample_data["clip_id"] = clip_id
                    sample_data["dataset"] = "{}_{}".format(data_name, sub_dir)

                    self.samples_total[sub_clip_id].append(sample_data)

        return clip_cnt, drop_frame_cnt

    def _normalize_pts(self, pts):
        """
        convert points to range [0, 1]
        :param pts:
        :return:
        """

        pts = pts[:, :2]
        pts[:, 0] = (pts[:, 0] - self.bev_range[0]) / (self.bev_range[1] - self.bev_range[0])  # x
        pts[:, 1] = (pts[:, 1] - self.bev_range[2]) / (self.bev_range[3] - self.bev_range[2])  # y
        return pts

    def _normalize_curb_attrs(self, attrs):
        """
        convert curb attributes to range [0, 1]
        :param attrs:
        :return:
        """

        # attrs[:, 0] = (attrs[:, 0] - self.attrs_range["class"][0]) / (
        #         self.attrs_range["class"][1] - self.attrs_range["class"][0])
        attrs[:, 0] = (attrs[:, 0] - self.attrs_range["curb_type"][0]) / (
                self.attrs_range["curb_type"][1] - self.attrs_range["curb_type"][0])
        attrs[:, 1] = (attrs[:, 1] - self.attrs_range["curb_subtype"][0]) / (
                self.attrs_range["curb_subtype"][1] - self.attrs_range["curb_subtype"][0])

        return attrs

    def _normalize_lane_attrs(self, attrs):
        """
        convert lane attributes to range [0, 1]
        :param attrs:
        :return:
        """

        # attrs[:, 0] = (attrs[:, 0] - self.attrs_range["class"][0]) / (
        #         self.attrs_range["class"][1] - self.attrs_range["class"][0])
        attrs[:, 0] = (attrs[:, 0] - self.attrs_range["lane_type"][0]) / (
                self.attrs_range["lane_type"][1] - self.attrs_range["lane_type"][0])
        attrs[:, 1] = (attrs[:, 1] - self.attrs_range["lane_color"][0]) / (
                self.attrs_range["lane_color"][1] - self.attrs_range["lane_color"][0])

        return attrs

    def _normalize_stop_line_attrs(self, attrs):
        """
        convert stop line attributes to range [0, 1]
        :param attrs:
        :return:
        """
        # attrs[:, 0] = (attrs[:, 0] - self.attrs_range["class"][0]) / (
        #         self.attrs_range["class"][1] - self.attrs_range["class"][0])

        return attrs

    def _convert_curb(self, curb_data, delta_ts=0.0, is_det=True):
        """
        convert detected curb data & gt curb data to the format for training
        :param curb_data:
        :param delta_ts:
        :return:
        """

        # in case gt has no curb
        if not curb_data or not curb_data['points']:
            return None, None, None

        if is_det:
            pad_pt_value = self.pad_pt_value_det
        else:
            pad_pt_value = self.pad_pt_value_gt

        # curb
        curbs_pt = []
        curbs_attr = []
        curbs_padding_flag = []
        # loop each curb element in curb data
        for idx, curb_pts in enumerate(curb_data["points"]):
            # points
            # normalize points to [0, 1]
            curb_pts = self._normalize_pts(curb_pts)
            if curb_pts.shape[0] > self.road_pt_pad_num:
                logger.error("curb points {} exceed the limit {}", curb_pts.shape[0], self.road_pt_pad_num)
                raise IndexError
            num_to_pad = self.road_pt_pad_num - curb_pts.shape[0]
            curbs_pt.append(np.pad(curb_pts, ((0, num_to_pad), (0, 0)), "constant", constant_values=pad_pt_value))

            # attributes
            attributes = np.stack([curb_data["type"][idx], curb_data["subtype"][idx]], axis=1)
            attributes = np.hstack(
                # [curb_type, curb_subtype, delta_ts]
                [attributes, np.full((attributes.shape[0], 1), delta_ts, dtype=np.float32), ]
            )
            attributes = self._normalize_curb_attrs(attributes)
            curbs_attr.append(
                np.pad(attributes, ((0, num_to_pad), (0, 0)), "constant", constant_values=self.pad_attr_value)
            )

            # padding flags
            padding_flag = np.full((curb_pts.shape[0]), 1)
            curbs_padding_flag.append(np.pad(padding_flag, (0, num_to_pad), "constant", constant_values=0))

        padded_curbs_pt = np.stack(curbs_pt)
        padded_curbs_attr = np.stack(curbs_attr)
        padding_flag = np.stack(curbs_padding_flag)

        return padded_curbs_pt, padded_curbs_attr, padding_flag

    def _convert_lane(self, lane_data, delta_ts=0.0, is_det=True):
        """
        convert detected lane data & gt lane data to the format for training
        :param lane_data:
        :param delta_ts:
        :return:
        """

        # in case gt has no lane
        if not lane_data or not lane_data['points']:
            return None, None, None

        if is_det:
            pad_pt_value = self.pad_pt_value_det
        else:
            pad_pt_value = self.pad_pt_value_gt

        # lane
        lanes_pt = []
        lanes_attr = []
        lanes_padding_flag = []
        # loop each lane element in lane data
        for idx, lane_pts in enumerate(lane_data["points"]):
            # points
            lane_pts = self._normalize_pts(lane_pts)
            if lane_pts.shape[0] > self.road_pt_pad_num:
                logger.error("lane points {} exceed the limit {}", lane_pts.shape[0], self.road_pt_pad_num)
                raise IndexError
            num_to_pad = self.road_pt_pad_num - lane_pts.shape[0]
            lanes_pt.append(np.pad(lane_pts, ((0, num_to_pad), (0, 0)), "constant", constant_values=pad_pt_value))

            # attributes
            attributes = np.stack([lane_data["type"][idx], lane_data["color"][idx]], axis=1)
            # add road element class flag
            attributes = np.hstack(
                # [type, color, delta_ts]
                [attributes, np.full((attributes.shape[0], 1), delta_ts, dtype=np.float32), ]
            )
            attributes = self._normalize_lane_attrs(attributes)
            lanes_attr.append(
                np.pad(attributes, ((0, num_to_pad), (0, 0)), "constant", constant_values=self.pad_attr_value)
            )

            # padding flags
            padding_flag = np.full((lane_pts.shape[0]), 1)
            lanes_padding_flag.append(np.pad(padding_flag, (0, num_to_pad), "constant", constant_values=0))

        padded_lanes_pt = np.stack(lanes_pt)
        padded_lanes_attr = np.stack(lanes_attr)
        padding_flag = np.stack(lanes_padding_flag)

        return padded_lanes_pt, padded_lanes_attr, padding_flag

    def _convert_stop_line(self, stop_line_data, delta_ts=0.0, is_det=True):
        """
        convert detected stop line data & gt stop line data to the format for training
        :param stop_line_data:
        :param delta_ts:
        :return:
        """

        # in case gt has no stop line
        if not stop_line_data or not stop_line_data['points']:
            return None, None, None

        if is_det:
            pad_pt_value = self.pad_pt_value_det
        else:
            pad_pt_value = self.pad_pt_value_gt

        stop_line_pt = []
        stop_line_attr = []
        stop_line_padding_flag = []
        # loop each stop line element in stop line data
        for idx, stop_line_pts in enumerate(stop_line_data["points"]):
            # points
            stop_line_pts = self._normalize_pts(stop_line_pts)
            if stop_line_pts.shape[0] > self.road_pt_pad_num:
                logger.error("stop line points {} exceed the limit {}", stop_line_pts.shape[0], self.road_pt_pad_num)
                raise IndexError
            num_to_pad = self.road_pt_pad_num - stop_line_pts.shape[0]
            stop_line_pt.append(
                np.pad(stop_line_pts, ((0, num_to_pad), (0, 0)), "constant", constant_values=pad_pt_value)
            )

            # add road element class flag
            attributes = np.hstack(
                # [delta_ts]
                [np.full((stop_line_pts.shape[0], 1), delta_ts, dtype=np.float32), ]
            )
            # attributes = self._normalize_stop_line_attrs(attributes)
            stop_line_attr.append(
                np.pad(attributes, ((0, num_to_pad), (0, 0)), "constant", constant_values=self.pad_attr_value)
            )

            # padding flags
            padding_flag = np.full((stop_line_pts.shape[0]), 1)
            stop_line_padding_flag.append(np.pad(padding_flag, (0, num_to_pad), "constant", constant_values=0))

        padded_stop_line_pt = np.stack(stop_line_pt)
        padded_stop_line_attr = np.stack(stop_line_attr)
        padding_flag = np.stack(stop_line_padding_flag)

        return padded_stop_line_pt, padded_stop_line_attr, padding_flag

    def _convert_road_data(self, road_data_samples, ts_list):
        """
        convert road detection data from argus_ppl to the format for training
        :param road_data_samples:
        :param ts_list:
        :return:
        """

        # history frames delta ts
        frames_ts = np.stack([ts for ts in ts_list])
        history_delta_ts = (frames_ts[-1] - frames_ts).astype(np.float32)

        road_element_pt_list = []
        road_element_attr_list = []
        road_element_pt_pad_flag_list = []
        road_element_pad_flag_list = []
        road_element_class_list = []
        for delta_ts, road_data in zip(history_delta_ts, road_data_samples):
            # curb
            padded_curbs_pt, padded_curbs_attr, curb_padding_flag = self._convert_curb(road_data["curbs"], delta_ts,
                                                                                       True)
            # lane
            padded_lanes_pt, padded_lanes_attr, lane_padding_flag = self._convert_lane(road_data["lanelines"], delta_ts,
                                                                                       True)
            # stop lines
            padded_stop_line_pt, padded_stop_line_attr, stop_line_padding_flag = self._convert_stop_line(
                road_data["stopline"], delta_ts, True
            )

            # aggregate
            # points
            pt_list = [item for item in [padded_curbs_pt, padded_lanes_pt, padded_stop_line_pt] if item is not None]
            padded_road_data_pt = np.vstack(pt_list).astype(np.float32)
            if padded_road_data_pt.shape[0] > self.road_element_pad_num:
                logger.error(
                    "road element numer {} exceed the limit {}", padded_road_data_pt.shape[0], self.road_element_pad_num
                )
                raise IndexError
            num_to_pad = self.road_element_pad_num - padded_road_data_pt.shape[0]
            road_element_pt_list.append(
                np.pad(
                    padded_road_data_pt,
                    ((0, num_to_pad), (0, 0), (0, 0)),
                    "constant",
                    constant_values=self.pad_pt_value_det,
                )
            )

            # padding flags
            padding_flag_list = [
                item for item in [curb_padding_flag, lane_padding_flag, stop_line_padding_flag] if item is not None
            ]
            pt_padded_flags = np.vstack(padding_flag_list)
            road_element_pt_pad_flag_list.append(
                np.pad(pt_padded_flags, ((0, num_to_pad), (0, 0)), "constant", constant_values=True)
            )
            element_padded_flags = np.full(pt_padded_flags.shape[0], False)
            road_element_pad_flag_list.append(
                np.pad(element_padded_flags, (0, num_to_pad), "constant", constant_values=True)
            )

            # attrs
            padded_road_data_class = np.full((self.road_element_pad_num,), self.pad_attr_value, dtype=np.float32)
            padded_road_data_attr = np.full(
                (self.road_element_pad_num, self.road_pt_pad_num, self.road_element_attr_dim),
                self.pad_attr_value,
                dtype=np.float32,
            )
            curb_sz = len(road_data["curbs"]["points"])
            lane_sz = len(road_data["lanelines"]["points"])
            stop_line_sz = len(road_data["stopline"]["points"])
            lane_end_idx = curb_sz + lane_sz
            stop_line_end_idx = curb_sz + lane_sz + stop_line_sz
            if curb_sz != 0:
                # assign curb attr
                padded_road_data_class[0:curb_sz] = self.class_normalize_value['curb']
                padded_road_data_attr[0:curb_sz, :, 0:2] = padded_curbs_attr[:, :, 0:2]
                padded_road_data_attr[0:curb_sz, :, 4] = padded_curbs_attr[:, :, 2]
            if lane_sz != 0:
                # assign lane class
                padded_road_data_class[curb_sz:lane_end_idx] = self.class_normalize_value['lane']
                # assign lane attr
                padded_road_data_attr[curb_sz:lane_end_idx, :, 2:4] = padded_lanes_attr[:, :, 0:2]
                padded_road_data_attr[curb_sz:lane_end_idx, :, 4] = padded_lanes_attr[:, :, 2]
            if stop_line_sz != 0:
                # assign stop line class
                padded_road_data_class[lane_end_idx:stop_line_end_idx] = self.class_normalize_value['stopline']
                padded_road_data_attr[lane_end_idx:stop_line_end_idx, :, 4] = padded_stop_line_attr[:, :, 0]

            road_element_attr_list.append(padded_road_data_attr)
            road_element_class_list.append(padded_road_data_class)

        return {
            "points": np.stack(road_element_pt_list),
            "attrs": np.stack(road_element_attr_list),
            "padding_pt_flag": np.stack(road_element_pt_pad_flag_list),
            "padding_element_flag": np.stack(road_element_pad_flag_list),
            "class": np.stack(road_element_class_list),
        }

    def _convert_gt(self, gt_samples):
        """
        convert 4dgt data to the format for training
        :param gt_samples:
        :return:
        """

        # curb
        padded_curbs_pt, padded_curbs_attr, curb_padding_flag = self._convert_curb(gt_samples["curbs"], is_det=False)
        curb_ids = [item for item in gt_samples["curbs"]["id"]] if gt_samples["curbs"] else None
        # lane
        padded_lanes_pt, padded_lanes_attr, lane_padding_flag = self._convert_lane(gt_samples["lanelines"],
                                                                                   is_det=False)
        lane_ids = [item for item in gt_samples["lanelines"]["id"]] if gt_samples["lanelines"] else None
        # stop lines
        padded_stop_line_pt, padded_stop_line_attr, stop_line_padding_flag = self._convert_stop_line(
            gt_samples["stopline"], is_det=False
        )
        stop_line_ids = [item for item in gt_samples["stopline"]["id"]] if gt_samples["stopline"] else None

        # aggregate
        # points
        pt_list = [item for item in [padded_curbs_pt, padded_lanes_pt, padded_stop_line_pt] if item is not None]
        padded_gt_pt = (np.vstack(pt_list)).astype(np.float32)

        # padding flags
        padding_flag_list = [
            item for item in [curb_padding_flag, lane_padding_flag, stop_line_padding_flag] if item is not None
        ]
        padded_flags = np.vstack(padding_flag_list)

        # ids
        id_list = [item for item in [curb_ids, lane_ids, stop_line_ids] if item is not None]
        gt_ids = np.hstack(id_list)

        # attributes
        curb_sz = 0 if not gt_samples["curbs"] else len(gt_samples["curbs"]["points"])
        lane_sz = 0 if not gt_samples["lanelines"] else len(gt_samples["lanelines"]["points"])
        stop_line_sz = 0 if not gt_samples["stopline"] else len(gt_samples["stopline"]["points"])
        element_num = curb_sz + lane_sz + stop_line_sz

        gt_class = np.full((element_num, 1), self.pad_attr_value, dtype=np.int64)
        padded_gt_curb_type = np.full(
            (curb_sz, self.road_pt_pad_num, 1),
            self.pad_attr_value,
            dtype=np.float32,
        )
        padded_gt_curb_subtype = np.full(
            (curb_sz, self.road_pt_pad_num, 1),
            self.pad_attr_value,
            dtype=np.float32,
        )
        padded_gt_lane_type = np.full(
            (lane_sz, self.road_pt_pad_num, 1),
            self.pad_attr_value,
            dtype=np.float32,
        )
        padded_gt_lane_color = np.full(
            (lane_sz, self.road_pt_pad_num, 1),
            self.pad_attr_value,
            dtype=np.float32,
        )

        lane_end_idx = curb_sz + lane_sz
        stop_line_end_idx = curb_sz + lane_sz + stop_line_sz
        if curb_sz != 0:
            # assign curb class
            gt_class[0:curb_sz, 0] = self.road_element_cls["curb"]
            # assign curb attr
            padded_gt_curb_type[:, :, 0] = padded_curbs_attr[:, :, 0]
            padded_gt_curb_subtype[:, :, 0] = padded_curbs_attr[:, :, 1]

        if lane_sz != 0:
            # assign lane class
            gt_class[curb_sz:lane_end_idx, 0] = self.road_element_cls["lane"]
            # assign lane attr
            padded_gt_lane_type[:, :, 0] = padded_lanes_attr[:, :, 0]
            padded_gt_lane_color[:, :, 0] = padded_lanes_attr[:, :, 1]
        if stop_line_sz != 0:
            # assign stop line class
            gt_class[lane_end_idx:stop_line_end_idx, 0] = self.road_element_cls["stopline"]

        if padded_gt_pt.shape[0] != gt_ids.shape[0]:
            raise RuntimeError("gt points and gt ids have different length!")

        return {
            "points": padded_gt_pt,
            "curb_type": padded_gt_curb_type,
            "curb_subtype": padded_gt_curb_subtype,
            "lane_type": padded_gt_lane_type,
            "lane_color": padded_gt_lane_color,
            "pt_padding_flag": padded_flags,
            "ids": gt_ids,
            "class": gt_class,
        }

    def _convert_sample(self, anno_data):
        """
        convert sample annotation data to training data
        :param anno_data:
        :return:
        """

        # load frame data from pickle
        sample_data = {"car_pose": [], "ts": [], "road_data": [], "lane_gt": []}
        for frame_path in anno_data["frame_path"]:
            if self.use_remote_data is False:
                if not os.path.exists(frame_path):
                    raise RuntimeError(f"frame path {frame_path} not exists!")
                with open(frame_path, "rb") as fin:
                    frame_data = pickle.load(fin)
            else:
                frame_data = pickle.loads(read_cache_op_py(frame_path)[0])

            # sample_data["car_pose"].append(frame_data["car_pose"])
            sample_data["car_pose"].append(np.zeros(1, dtype=np.float32))
            sample_data["road_data"].append(frame_data["road_data"])
            sample_data["lane_gt"].append(frame_data["lane_gt"])
            if self.use_remote_data is False:
                ts = float(frame_path.split("/")[-1][:-4])
            else:
                ts = float(frame_path[0].split(" ")[0].split("/")[-1][:-4])
            sample_data["ts"].append(ts)
        sample_ret = copy.deepcopy(anno_data)

        # pose
        sample_ret["car_pose"] = np.stack(sample_data["car_pose"], axis=0).astype(np.float32)

        # timestamp
        sample_ret["ts"] = np.array(sample_data["ts"]).astype(np.float64)
        sample_ret["cur_ts"] = sample_data["ts"][-1]

        # road data
        sample_ret["road_data"] = self._convert_road_data(sample_data["road_data"], sample_data["ts"])

        # gt
        sample_ret["lane_gt"] = self._convert_gt(sample_data["lane_gt"][-1])

        return sample_ret

    def parse_one_dataset(self, data_name, data_meta):
        """
        parse one dataset from configuration
        :param data_name: dataset name
        :param data_meta: dataset meta data
        :return:
        """

        # collect all annotation files dirs
        if self.use_remote_data is False:
            sub_dirs = data_meta["sub_dirs"]
            anno_files = []
            for item in sub_dirs:
                anno_dir = os.path.join(data_meta["root_dir"], item, "annotation.dat")
                anno_files.append(anno_dir)
        else:
            anno_files = data_meta["meta_data"]

        # parse all annotation files; generate training samples
        for anno_path in anno_files:
            if self.use_remote_data is False:
                anno_clip_cnt, anno_drop_frame_cnt = self._parse_single_anno(anno_path, data_meta["root_dir"], data_name)
            else:
                anno_clip_cnt, anno_drop_frame_cnt = self._parse_single_anno(anno_path, None, data_name)
            self.total_clip_cnt += anno_clip_cnt
            self.total_drop_frame_cnt += anno_drop_frame_cnt

    def _sample_indices(self, sub_clip_id, start_idx):
        """
        generate sample indices for a sub clip
        :param sub_clip_id: id of the sub clip
        :param start_idx: index of start sample
        :return:
        """
        # assert this sub clip has enough frames to sample
        max_idx = self.sub_clip_max_idx[sub_clip_id]
        assert max_idx - start_idx >= self.sub_clip_sample_size - 1

        # get a random sample interval
        # rand_sample_interval = randint(1, self.sample_interval + 1)

        # get sample indices
        ids = [start_idx + self.sample_interval * i for i in range(self.sub_clip_sample_size)]
        return [min(i, max_idx) for i in ids], self.sample_interval

    @staticmethod
    def _reassign_sub_clip_gt_id(sub_clip_data_samples):
        """
        original gt ids are str, reassign them to int
        :param sub_clip_data_samples:
        :return:
        """

        all_gt_id_set = set()
        for sample in sub_clip_data_samples:
            sample_gt_id_set = set()
            for gt_id in sample["lane_gt"]["ids"]:
                if gt_id in sample_gt_id_set:
                    raise RuntimeError("gt id {} already exists in sample".format(gt_id))
                sample_gt_id_set.add(gt_id)
            all_gt_id_set.update(sample_gt_id_set)

        gt_id_map = {}
        for idx, gt_id in enumerate(all_gt_id_set):
            gt_id_map[gt_id] = idx

        for sample in sub_clip_data_samples:
            sample["lane_gt"]["ids"] = np.array([gt_id_map[gt_id] for gt_id in sample["lane_gt"]["ids"]])

    def __getitem__(self, idx: int):
        """
        get training item
        :param idx: index
        :return: training item
        """

        sub_clip_id, start_idx = self.video_sub_clips[idx]
        # get sample indices from sub clip
        indices, sample_interval = self._sample_indices(sub_clip_id, start_idx)
        # get samples from indices
        sub_clip_anno_samples = [self.samples_total[sub_clip_id][i] for i in indices]
        # convert samples from annotation to training data
        sub_clip_data_samples = [self._convert_sample(anno_sample) for anno_sample in sub_clip_anno_samples]

        # reassign gt ids from str to int
        self._reassign_sub_clip_gt_id(sub_clip_data_samples)

        ret_item = {}
        ret_item["id"] = idx
        ret_item["sample_interval"] = sample_interval
        ret_item["sub_clip_id"] = sub_clip_id
        ret_item["dataset"] = sub_clip_data_samples[0]["dataset"]
        ret_item["car_pose"] = [sample["car_pose"] for sample in sub_clip_data_samples]
        ret_item["road_data"] = [sample["road_data"] for sample in sub_clip_data_samples]
        ret_item["lane_gt"] = [sample["lane_gt"] for sample in sub_clip_data_samples]
        ret_item["history_ts"] = [sample["ts"] for sample in sub_clip_data_samples]
        ret_item["sub_clip_ts"] = [sample["cur_ts"] for sample in sub_clip_data_samples]

        # data augmentation
        if self._transforms is not None:
            ret_item = self._transforms(ret_item)

        return ret_item

    def batch_collate(self, data: List[Any]) -> Any:
        """
        aggregate batch data; from numpy to tensor
        :param data:
        :return:
        """

        batch_car_pose = []
        batch_road_data_points = []
        batch_road_data_class = []
        batch_road_data_attrs = []
        batch_road_data_padding_pt_flags = []
        batch_road_data_padding_element_flags = []

        batch_gt_points = []
        batch_gt_curb_types = []
        batch_gt_curb_subtypes = []
        batch_gt_lane_types = []
        batch_gt_lane_colors = []
        batch_gt_pt_padding_flags = []
        batch_gt_ids = []
        batch_gt_class = []
        batch_gt_num = []

        batch_ts = []
        batch_cur_ts = np.stack([data_entry["sub_clip_ts"] for data_entry in data])
        batch_delta_ts = []

        # avoid duplicate gt ids in different batch
        max_id = 0
        for batch_data in data:
            for idx in range(self.sub_clip_sample_size):
                max_id = max(max_id, np.max(batch_data["lane_gt"][idx]["ids"]))
        gt_id_interval = max_id + 1
        for batch_idx, batch_data in enumerate(data):
            for idx in range(self.sub_clip_sample_size):
                batch_data["lane_gt"][idx]["ids"] += gt_id_interval * batch_idx

        # aggregate batch data in to training sub clips
        for idx in range(self.sub_clip_sample_size):
            batch_car_pose.append(np.stack([data_entry["car_pose"][idx] for data_entry in data]))

            batch_road_data_points.append(np.stack([data_entry["road_data"][idx]["points"] for data_entry in data]))
            batch_road_data_attrs.append(np.stack([data_entry["road_data"][idx]["attrs"] for data_entry in data]))
            batch_road_data_class.append(np.stack([data_entry["road_data"][idx]["class"] for data_entry in data]))
            batch_road_data_padding_pt_flags.append(
                np.stack([data_entry["road_data"][idx]["padding_pt_flag"] for data_entry in data])
            )
            batch_road_data_padding_element_flags.append(
                np.stack([data_entry["road_data"][idx]["padding_element_flag"] for data_entry in data])
            )

            batch_gt_points.append(np.vstack([data_entry["lane_gt"][idx]["points"] for data_entry in data]))
            batch_gt_curb_types.append(np.vstack([data_entry["lane_gt"][idx]["curb_type"] for data_entry in data]))
            batch_gt_curb_subtypes.append(
                np.vstack([data_entry["lane_gt"][idx]["curb_subtype"] for data_entry in data])
            )
            batch_gt_lane_types.append(np.vstack([data_entry["lane_gt"][idx]["lane_type"] for data_entry in data]))
            batch_gt_lane_colors.append(np.vstack([data_entry["lane_gt"][idx]["lane_color"] for data_entry in data]))
            batch_gt_pt_padding_flags.append(
                np.vstack([data_entry["lane_gt"][idx]["pt_padding_flag"] for data_entry in data])
            )
            batch_gt_class.append(np.vstack([data_entry["lane_gt"][idx]["class"] for data_entry in data]))
            batch_gt_ids.append(np.hstack([data_entry["lane_gt"][idx]["ids"] for data_entry in data]))
            batch_gt_num.append(
                np.array([data_entry["lane_gt"][idx]["points"].shape[0] for data_entry in data]).astype(np.int32)
            )

            batch_samples_ts = np.stack([data_entry["history_ts"][idx] for data_entry in data])
            batch_ts.append(batch_samples_ts)
            batch_delta_ts.append((batch_samples_ts[:, -1:] - batch_samples_ts).astype(np.float32))

        # final batch items
        batch_items = {
            "sub_clip_id": [sample_data["sub_clip_id"] for sample_data in data],
            "id": [sample_data["id"] for sample_data in data],
            "car_poses": [torch.from_numpy(car_pose) for car_pose in batch_car_pose],
            "road_gt": {
                "gt_points": [torch.from_numpy(gt_point) for gt_point in batch_gt_points],
                # "gt_curb_types": [torch.from_numpy(gt_curb_type) for gt_curb_type in batch_gt_curb_types],
                # "gt_curb_subtypes": [torch.from_numpy(gt_curb_subtype) for gt_curb_subtype in batch_gt_curb_subtypes],
                # "gt_lane_types": [torch.from_numpy(gt_lane_type) for gt_lane_type in batch_gt_lane_types],
                # "gt_lane_colors": [torch.from_numpy(gt_lane_color) for gt_lane_color in batch_gt_lane_colors],
                "gt_pt_padding_flags": [
                    torch.from_numpy(gt_padding_flag) for gt_padding_flag in batch_gt_pt_padding_flags
                ],
                "gt_ids": [torch.from_numpy(gt_ids) for gt_ids in batch_gt_ids],
                "gt_class": [torch.from_numpy(gt_class) for gt_class in batch_gt_class],
                "gt_num": [torch.from_numpy(gt_num) for gt_num in batch_gt_num],
            },
            "road_data_points": [
                torch.from_numpy(road_data_points).to(torch.float32) for road_data_points in batch_road_data_points
            ],
            "road_data_class": [
                torch.from_numpy(road_data_class).to(torch.float32) for road_data_class in batch_road_data_class
            ],
            # "road_data_attrs": [
            #     torch.from_numpy(road_data_attrs).to(torch.float32) for road_data_attrs in
            #     batch_road_data_attrs
            # ],
            "road_data_padding_pt_flags": [
                torch.from_numpy(road_data_padding_flags)
                for road_data_padding_flags in batch_road_data_padding_pt_flags
            ],
            "road_data_padding_element_flags": [
                torch.from_numpy(road_data_padding_flags)
                for road_data_padding_flags in batch_road_data_padding_element_flags
            ],
            "history_delta_ts": [torch.from_numpy(delta_ts) for delta_ts in batch_delta_ts],
            "history_ts": batch_ts,
            "sub_clip_ts": batch_cur_ts,
        }

        return batch_items



def build_dataset(image_set, args):
    root = Path(args.data_path)
    assert root.exists(), f'provided Lane path {root} does not exist'
    if image_set == 'train':
        data_txt_path = args.data_txt_path_train
        dataset = LaneDataset()
    if image_set == 'val':
        data_txt_path = args.data_txt_path_val
        dataset = LaneDataset()
    return dataset
