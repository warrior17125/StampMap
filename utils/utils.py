# -*- coding: utf-8 -*-
import itertools
from typing import Any, Dict, List, Union
import torch


class StampMapTensorboardManager():
    def __init__(self,
                 log_dir: str = "./",
                 comment: str = "",
                 filename_suffix: str = "",
                 max_queue: int = 10,
                 flush_secs: int = 120,
                 rank: int = 0, ):
        super().__init__(log_dir, comment, filename_suffix, max_queue, flush_secs, rank)

    def log_scaler_eval(self, value_list: Dict, global_epoch: int):
        for key, value in value_list.items():
            tag = "eval/" + key
            self._writer.add_scalar(tag, value, global_epoch)


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Instances:
    def __init__(self, instance_type, **kwargs: Any):
        self._instance_type = instance_type
        if self._instance_type not in ["gt", "track"]:
            raise NotImplementedError("type must be 'gt' or 'track'!")

        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def instance_type(self) -> str:

        return self._instance_type

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances!".format(name))
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        """
        # data_len = len(value)
        # if len(self._fields):
        #     assert (
        #             len(self) == data_len
        #     ), "Adding a field of length {} to a Instances of length {}".format(data_len, len(self))
        self._fields[name] = value

    def has(self, name: str) -> bool:
        """
        Returns:
            bool: whether the field called `name` exists.
        """
        return name in self._fields

    def remove(self, name: str) -> None:
        """
        Remove the field called `name`.
        """
        del self._fields[name]

    def get(self, name: str) -> Any:
        """
        Returns the field called `name`.
        """
        return self._fields[name]

    def get_fields(self) -> Dict[str, Any]:
        """
        Returns:
            dict: a dict which maps names (str) to data of the fields

        Modifying the returned dict will modify this instance.
        """
        return self._fields

    # Tensor-like methods
    def to(self, *args: Any, **kwargs: Any) -> "Instances":
        """
        Returns:
            Instances: all fields are called with a `to(device)`, if the field has this method.
        """
        ret = Instances(self._instance_type)
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            ret.set(k, v)
        return ret

    def numpy(self):
        ret = Instances(self._instance_type)
        for k, v in self._fields.items():
            if hasattr(v, "numpy"):
                v = v.numpy()
            ret.set(k, v)
        return ret

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Instances":
        """
        Args:
            item: an index-like object and will be used to index all the fields.

        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Instances` where all fields are indexed by `item`.
        """
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("Instances index out of range!")
            else:
                item = slice(item, None, len(self))

        ret = Instances(self._instance_type)

        for k, v in self._fields.items():
            if k == "batch_num":
                batch_cum_sum = torch.cat(
                    (torch.zeros(1, device=v.device, dtype=torch.int32), torch.cumsum(v, dim=0))
                )

                if item.dtype == torch.bool:
                    batch_num = torch.zeros(batch_cum_sum.size(0) - 1, dtype=torch.int64, device=v.device)
                    for i, (start, end) in enumerate(zip(batch_cum_sum[:-1], batch_cum_sum[1:])):
                        batch_num[i] = item[start:end].sum()
                else:
                    bool_tensor = torch.logical_and(
                        torch.ge(item.unsqueeze(-1), batch_cum_sum[:-1]),
                        torch.lt(item.unsqueeze(-1), batch_cum_sum[1:]),
                    )
                    batch_num = torch.sum(bool_tensor, dim=0)
                ret.set(k, batch_num)
            else:
                ret.set(k, v[item])

        return ret

    def __len__(self) -> int:
        for k, v in self._fields.items():
            # gt
            if k == "labels":
                return v.shape[0]
            # pred
            if k == "query_embed":
                if len(v.shape) == 3:
                    return v.shape[0] * v.shape[1]
                elif len(v.shape) == 2:
                    return v.shape[0]

        raise NotImplementedError("Empty Instances does not support __len__!")

    def __iter__(self):
        raise NotImplementedError("`Instances` object is not iterable!")

    @staticmethod
    def cat(instance_lists: List["Instances"]) -> "Instances":
        """
        Args:
            instance_lists (list[Instances])

        Returns:
            Instances
        """
        assert all(isinstance(i, Instances) for i in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]

        instance_type = instance_lists[0].instance_type
        ret = Instances(instance_type)
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            else:
                raise ValueError("Unsupported type {} for concatenation".format(type(v0)))
            ret.set(k, values)
        return ret

    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "type={}, ".format(self._instance_type)
        s += "num_instances={}, ".format(len(self))
        s += "fields=[{}])".format(", ".join((f"{k}: {v}" for k, v in self._fields.items())))
        return s

    __repr__ = __str__


def random_drop_tracks(track_instances: Instances, drop_probability: float) -> Instances:
    if drop_probability > 0 and len(track_instances) > 0:
        keep_idxes = torch.rand_like(track_instances.scores) > drop_probability
        track_instances = track_instances[keep_idxes]
    return track_instances
