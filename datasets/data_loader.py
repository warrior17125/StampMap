import logging
from typing import Any
from typing import Dict
from typing import Optional

from torch.utils.data import DataLoader

class LaneDataLoader(DataLoader):
    """Stamp Torchpilot DataLoader."""

    # pylint: disable=super-init-not-called.
    def __init__(
        self,
        dataset_cfg: Dict[str, Any],
        sampler_cfg: Dict[str, Any],
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
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
            raise ValueError(f"Expect type(dataset)=BaseDataset but got {type(dataset)}.")
        collate_fn = dataset.batch_collate
        if not callable(collate_fn):
            raise ValueError("Expect callable(collate_fn)=True but got False.")

        # Fake batch size.
        task_batch_size = batch_size
        if isinstance(batch_size, dict):
            batch_size = 1

        sampler_cfg["dataset"] = dataset
        sampler_cfg["batch_size"] = task_batch_size
        sampler = SAMPLER.build(sampler_cfg)
        if not isinstance(sampler, TPDistributedSampler):
            raise ValueError(f"Expect type(sampler)=TPDistributedSampler but got {type(sampler)}.")

        if sampler.drop_last != drop_last:
            raise ValueError("Inconsistent `drop_last` in sampler and dataloader.")

        DataLoader.__init__(
            self,
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
        # self._data_iter = iter(self)
        self._step = 0
        self._to_cuda = to_cuda
        self._recursive_to_cuda = recursive_to_cuda
        self._pin_memory = pin_memory