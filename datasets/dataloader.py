# -*- coding: utf-8 -*-
"""Torchpilot DataLoader."""

import logging
import math
import os
import random
from functools import partial
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from torch.utils.data import DataLoader
from dataset import LaneDataset
import numpy as np
import torch
from torch import distributed as dist


logger = logging.getLogger(__name__)


class LaneDataLoader(DataLoader):
    def __init__(
        self,
        dataset: LaneDataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        sampler = None,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        multiprocessing_context=None,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        seed: Optional[int] = None,
        to_cuda: Optional[bool] = False,
        recursive_to_cuda: Optional[bool] = False,
    ):
        """Init."""
        dataset = DATASET.build(dataset_cfg)
        if not isinstance(dataset, BaseDataset):
            raise ValueError(
                f"Expect type(dataset)=BaseDataset but got {type(dataset)}."
            )
        collate_fn = dataset.batch_collate
        if not callable(collate_fn):
            raise ValueError("Expect callable(collate_fn)=True but got False.")

        dataset.send_data_cost()

        # Fake batch size.
        task_batch_size = batch_size
        if isinstance(batch_size, dict):
            batch_size = 1

        sampler_cfg["dataset"] = dataset
        sampler_cfg["batch_size"] = task_batch_size
        sampler = SAMPLER.build(sampler_cfg)
        if not isinstance(sampler, TPDistributedSampler):
            raise ValueError(
                f"Expect type(sampler)=TPDistributedSampler but got {type(sampler)}."
            )

        if sampler.drop_last != drop_last:
            raise ValueError("Inconsistent `drop_last` in sampler and dataloader.")

        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=_init_worker_funcion(workers_per_gpu=1, seed=seed),
            multiprocessing_context=multiprocessing_context,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
        self._data_iter = None  # Init to None
        self._step = 0
        self._to_cuda = to_cuda
        self._recursive_to_cuda = recursive_to_cuda
        self._pin_memory = pin_memory

    def get_sample_per_epoch(self) -> int:
        """Get sample per epoch."""
        indices = self.sampler.get_indices()  # type: ignore[attr-defined]
        return len(indices) if indices else len(self.dataset)

    def get_sample_num_per_gpu(self) -> int:
        """Get num samples."""
        return len(self.sampler)  # type: ignore[arg-type]

    def get_batch_size_per_gpu(self) -> Optional[int]:
        """Get batch size per gpu."""
        return self.batch_size

    def get_one_epoch_step_per_gpu(self) -> int:
        """Get one epoch step per gpu."""
        if self.batch_size is None:
            raise ValueError("Expect self.batch_size is not None, but got None.")
        data_per_gpu = len(self.sampler)  # type: ignore[arg-type]
        if not self.drop_last and data_per_gpu % self.batch_size != 0:
            return math.ceil(data_per_gpu / self.batch_size)
        return int(data_per_gpu / self.batch_size)

    def get_state(self) -> Tuple[int, int, List[int]]:
        """Get dataloader state. i.e. `epoch`, `step`, `sampler.indices`."""
        return (
            self.sampler.get_epoch(),  # type: ignore[attr-defined]
            self._step,
            self.sampler.get_indices(),  # type: ignore[attr-defined]
        )

    def set_state(
        self,
        epoch: int,
        step: int = 0,
        indices: Optional[List[int]] = None,
    ):
        """Set state.

        Args:
            epoch (int): The epoch.
            step (int): The step.
                Default: `0`.
            indices (List[int], optional): If set, will use this instead of the default
                `list(range(len(self.dataset)))` of sampler.
                Default: `None`.
        """
        if not isinstance(self.sampler, TPDistributedSampler):
            raise ValueError(
                f"Expect type(sampler)=`TPDistributedSampler` but got "
                f"type(sampler)={type(self.sampler)}."
            )

        # Epoch.
        self.sampler.set_epoch(epoch)

        # Step.
        self.sampler.set_step(step)

        # Indices.
        if indices is not None:
            self.sampler.set_indices(indices)

        # Reset.
        self.reset(step)

    def reset(self, step: int = 0) -> None:
        """Reset data iter."""
        self._data_iter = iter(self)  # type: ignore[assignment]
        self._step = step

    def recursive_to_cuda(self, data_to_recursive):
        """Move data to cuda recursively.

        Args:
            data_to_recursive: Data to move to cuda recursively.

        Returns:
            data_to_recursive: Data on cuda.
        """
        for key, data_item in data_to_recursive.items():
            if isinstance(data_item, torch.Tensor):
                if self._pin_memory:
                    data_item = data_item.pin_memory()
                data_to_recursive[key] = data_item.cuda(non_blocking=True)
                del data_item
            elif isinstance(data_item, dict):
                self.recursive_to_cuda(data_item)
            elif isinstance(data_item, (list, tuple)):
                for data_in in data_item:
                    self.recursive_to_cuda(data_in)
        return data_to_recursive

    def move_data_to_cuda(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Move data to cuda.

        Args:
            data Dict[str, Any]: Data to move.

        Returns:
            data Dict[str, Any]: Data on cuda.
        """
        if self._to_cuda:  # pylint: disable=too-many-nested-blocks
            for key, data_item in data.items():
                if isinstance(data_item, torch.Tensor):
                    if self._pin_memory and data_item.device.type != "cuda":
                        data_item = data_item.pin_memory()
                    data[key] = data_item.cuda(non_blocking=True)
                    del data_item
                elif isinstance(data_item, (list, tuple)):
                    for idx, list_item in enumerate(data_item):
                        if isinstance(list_item, torch.Tensor):
                            if self._pin_memory and list_item.device.type != "cuda":
                                list_item = list_item.pin_memory()
                            data[key][idx] = list_item.cuda(non_blocking=True)
                            del list_item
        elif self._recursive_to_cuda:
            data = self.recursive_to_cuda(data)

        return data

    def get_batch_data(self) -> Dict[str, Any]:
        """Get one batch data.

        Returns:
            One batch Data. Dict[str, Any].
        """
        if self._data_iter is None:
            raise ValueError(
                "dataloader.set_state() is expected before get batch data."
            )
        try:
            data = next(self._data_iter)
        except StopIteration:
            self.reset()
            data = next(self._data_iter)

        if self._to_cuda or self._recursive_to_cuda:
            if not isinstance(data, dict):
                raise ValueError(f"Expect type(data)=dict, but got {type(data)}.")

        data = self.move_data_to_cuda(data)

        self._step += 1
        return data


def _init_worker_funcion(
    workers_per_gpu: int = 1,
    seed: Optional[int] = None,
) -> Any:
    """Initialize worker function.

    Args:
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU, generally as random seed.
        seed (int, Optional): Seed to be used. Default: None.

    Returns:
        Callable function.
    """
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0

    init_fn = (
        partial(_worker_init_fn, num_workers=workers_per_gpu, rank=rank, seed=seed)
        if seed is not None
        else None
    )

    return init_fn


def _worker_init_fn(
    worker_id: int,
    num_workers: int,
    rank: int,
    seed: int,
) -> None:
    """Worker init function.

    Args:
        worker_id (int): The worker id.
        num_workers (int): Worker number.
        rank (int): The process rank.
        seed (int): Seed to be used.
    """
    affinity = os.sched_getaffinity(0)
    logger.warning(
        "GPU Process %d DataLoaderWorker %d set affinity to: %s",
        rank,
        worker_id,
        affinity,
    )
    # The seed of each worker equals to (num_worker * rank + worker_id + user_seed)
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
