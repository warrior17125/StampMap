import torch
from torch import nn
from utils.utils import Instances

class TrackQueryLifeManager():
    """
    track query life management
    Drived from MOTR v2
    https://github.com/megvii-research/MOTRv2/blob/main/models/qim.py
    """

    def __init__(self, score_thresh, filter_score_thresh, miss_tolerance, infer_batch_size, road_element_cls):
        super().__init__()
        self._score_thresh = score_thresh
        self._background_class = road_element_cls['background']

        # infer
        self._filter_score_thresh = filter_score_thresh
        self._miss_tolerance = miss_tolerance
        self._infer_batch_size = infer_batch_size
        self._max_obj_id = 0

    def clear(self):
        self._max_obj_id = 0

    def _infer_update_id(self, track_instances: Instances):
        for sample_idx in range(self._infer_batch_size):
            device = track_instances.matched_gt_ids[sample_idx].device

            # reset disappear time when track score > score_thresh
            track_instances.disappear_time[track_instances.track_score[sample_idx] >= self._score_thresh] = 0
            # collect new track queries
            # filter out background predictions
            class_pred_mask = (track_instances.cls_pred.softmax(-1).argmax(-1) != self._background_class).squeeze(0)
            new_obj = (track_instances.matched_gt_ids[sample_idx] == -1) & (
                    track_instances.track_score[sample_idx] >= self._score_thresh) & class_pred_mask
            # collect disappeared track queries
            disappeared_obj = (track_instances.matched_gt_ids[sample_idx] >= 0) & (
                    track_instances.track_score[sample_idx] < self._filter_score_thresh)

            # assign new object ids
            num_new_objs = new_obj.sum().item()
            track_instances.matched_gt_ids[sample_idx][new_obj] = self._max_obj_id + torch.arange(num_new_objs,
                                                                                                  device=device)
            self._max_obj_id += num_new_objs

            # mark disappeared objects
            track_instances.disappear_time[disappeared_obj] += 1
            to_del = disappeared_obj & (track_instances.disappear_time >= self._miss_tolerance)
            track_instances.matched_gt_ids[sample_idx][to_del] = -1

        return track_instances

    @staticmethod
    def _flatten_track_instance(track_instances):
        """
        restore from (batch_size, num_queries) to batch_size * num_queries
        :param track_instances:
        :return:
        """
        track_instances.query_embed = track_instances.query_embed.flatten(0, 1)
        track_instances.matched_gt_ids = track_instances.matched_gt_ids.flatten(0, 1)
        track_instances.track_score = track_instances.track_score.flatten(0, 1)
        track_instances.cls_pred = track_instances.cls_pred.flatten(0, 1)
        track_instances.point_confidence_pred = track_instances.point_confidence_pred.flatten(0, 1)
        track_instances.point_coord_pred = track_instances.point_coord_pred.flatten(0, 1)
        track_instances.query_indices = track_instances.query_indices.flatten(0, 1)

        return track_instances

    def train_forward(self, track_instances) -> Instances:
        flattened_track_instances = self._flatten_track_instance(track_instances)

        active_idxes = flattened_track_instances.matched_gt_ids >= 0
        # filter out background predictions
        # class_pred_mask = flattened_track_instances.cls_pred.softmax(-1).argmax(-1) != self._background_class
        # active_idxes = active_idxes & class_pred_mask
        active_track_instances = flattened_track_instances[active_idxes]

        return active_track_instances

    def infer_forward(self, track_instances) -> Instances:
        track_instances = self._infer_update_id(track_instances)
        flattened_track_instances = self._flatten_track_instance(track_instances)
        active_track_instances = flattened_track_instances[flattened_track_instances.matched_gt_ids >= 0]

        return active_track_instances
    

class TrackQueriesGenerator():
    """
    generate empty track queries
    """

    def __init__(self, num_queries, d_model, batch_size, road_class_num, road_pt_num):
        super().__init__()
        self.query_embed = nn.Embedding(num_queries, d_model)
        self.num_queries = num_queries
        self.d_model = d_model
        self.batch_size = batch_size
        self.road_class_num = road_class_num
        self.road_pt_num = road_pt_num

    def set_mode(self, train_mode: bool = True, ) -> None:
        self._train_mode = train_mode
        if not self._train_mode:
            self.batch_size = 1

    def _generate_empty_tracks(self):
        track_instances = Instances("track")
        device = self.query_embed.weight.device
        track_instances.query_embed = self.query_embed.weight
        track_instances.matched_gt_ids = torch.full((self.num_queries,), -1, dtype=torch.long, device=device)
        track_instances.track_score = torch.zeros(self.num_queries, dtype=torch.float32, device=device)
        track_instances.query_indices = torch.arange(self.num_queries, dtype=torch.int64, device=device)
        track_instances.batch_num = torch.tensor([self.num_queries], dtype=torch.int64, device=device)
        if not self._train_mode:
            track_instances.disappear_time = torch.zeros(self.num_queries, dtype=torch.float32, device=device)

        return track_instances.to(device)

    def _restore_batch_dim(self, track_instances):
        """
        restore from batch_size * num_queries to (batch_size, num_queries)
        :param track_instances:
        :return:
        """
        track_instances.query_embed = track_instances.query_embed.view(self.batch_size, self.num_queries, self.d_model)
        track_instances.matched_gt_ids = track_instances.matched_gt_ids.view(self.batch_size, self.num_queries)
        track_instances.track_score = track_instances.track_score.view(self.batch_size, self.num_queries)
        track_instances.query_indices = track_instances.query_indices.view(self.batch_size, self.num_queries)

        return track_instances

    def common_forward(self, track_instances):
        """
        generate empty track queries to match self.num_queries
        :param track_instances:
        :return:
        """

        if track_instances is None:
            track_instances = Instances.cat([self._generate_empty_tracks() for _ in range(self.batch_size)])
        else:
            device = track_instances.query_embed.device
            # number of left queries for each sample in a batch
            batch_cum_sum = torch.cat(
                (torch.zeros(1, device=device, dtype=torch.int32),
                 torch.cumsum(track_instances.batch_num, dim=0))
            )
            track_instances.remove("batch_num")

            batch_track_instances = []
            # loop samples in a batch
            for sample_idx in range(self.batch_size):
                # for all the preset track queries, select the ones not in current track instances
                empty_instance = self._generate_empty_tracks()
                keep_track_indices = track_instances.query_indices[
                                     batch_cum_sum[sample_idx]:batch_cum_sum[sample_idx + 1]]
                empty_instance_indices = torch.full((self.num_queries,), True, dtype=torch.bool, device=device)
                # empty_instance_indices[keep_track_indices] = False
                for keep_track_index in keep_track_indices:
                    empty_instance_indices[keep_track_index] = False
                new_track_instance = empty_instance[empty_instance_indices]
                # get the existed track queries in current track instances
                keep_sample_idx = torch.arange(batch_cum_sum[sample_idx], batch_cum_sum[sample_idx + 1],
                                               device=device, dtype=torch.long)
                keep_instance = track_instances[keep_sample_idx]
                # concat the existed track queries and the new track queries as the final track queries
                new_sample_instance = Instances.cat([keep_instance, new_track_instance])
                new_sample_instance.batch_num = torch.tensor([self.num_queries], dtype=torch.int64,
                                                             device=device)
                batch_track_instances.append(new_sample_instance)
            track_instances = Instances.cat([sample for sample in batch_track_instances])

        # restore the batch dimension
        track_instances = self._restore_batch_dim(track_instances)
        return track_instances

    def train_forward(self, track_instances):
        return self.common_forward(track_instances)

    def infer_forward(self, track_instances):
        return self.common_forward(track_instances)
    

def build_lifemanager(args):
    return TrackQueryLifeManager(
        score_thresh=args.query_score_threshold, 
        filter_score_thresh=args.query_filter_score_threshold, 
        miss_tolerance=args.miss_tolerance, 
        infer_batch_size=args.infer_batch_size, 
        road_element_cls=args.road_element_cls,
        )

def build_generator(args):
    return TrackQueriesGenerator(
        num_queries=args.num_track_query, 
        d_model=args.dim_model,
        batch_size=args.train_batch_size,
        road_class_num=len(args.road_element_cls),
        road_pt_num=args.road_pt_pad_num
        )