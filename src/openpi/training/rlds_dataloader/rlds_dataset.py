"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import numpy as np
import torch
from PIL import Image
from torch.utils.data import IterableDataset
from openpi.training.rlds_dataloader.dataset import make_interleaved_dataset
from typing import List
from openpi.training.rlds_dataloader.materialize import get_oxe_dataset_kwargs_and_weights

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100
WORKER_SCALE_FACTOR = 4

class RLDSDataset(IterableDataset):
    def __init__(
        self,
        data_root_dir: Path,
        mixture_spec: List[Tuple[str, float]],
        action_key_list: List[str],
        batch_size: int,
        num_workers: int,
        action_horizon: int,
        valid_episodes: List[int] = [],
        shuffle_buffer_size: int = 256_000,
        train: bool = True,
    ) -> None:
        """Lightweight wrapper around RLDS TFDS Pipeline for use with PyTorch/OpenVLA Data Loaders."""
        self.data_root_dir = data_root_dir
        self.action_horizon = action_horizon
        self.batch_size = batch_size
        self.num_workers = num_workers

        # fmt: off
        per_dataset_kwargs, weights = get_oxe_dataset_kwargs_and_weights(
            self.data_root_dir,
            mixture_spec,
            action_key_list=action_key_list,
            load_camera_views=("primary", "secondary", "wrist"),
            load_depth=False,
            load_proprio=True,
            load_language=True,
            # action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
        )

        
        ## TODO(YY): MAKE sure action chunking is correct
        rlds_config = dict(
            # batch_size=self.batch_size,
            traj_transform_kwargs=dict(
                window_size=1,                                      # If we wanted to feed / predict more than one step
                future_action_window_size=self.action_horizon - 1,                        # For action chunking
                skip_unlabeled=False,                                # Skip trajectories without language labels
                goal_relabeling_strategy=None,                 # Goals are currently unused
            ),
            frame_transform_kwargs=dict(
                num_parallel_calls=self.num_workers,                       # For CPU-intensive ops (decoding, resizing, etc.)
            ),
            dataset_kwargs_list=per_dataset_kwargs,
            shuffle_buffer_size=shuffle_buffer_size,
            sample_weights=weights,
            balance_weights=False,
            # traj_transform_threads= self.num_workers * len(mixture_spec),
            # traj_read_threads= WORKER_SCALE_FACTOR * len(mixture_spec),
            traj_transform_threads=self.num_workers,
            traj_read_threads=self.num_workers,
            train=train,
        )
        # fmt: on
        # Initialize RLDS Dataset
        self.dataset, self.dataset_length, self.dataset_statistics = self.make_dataset(rlds_config)

    def make_dataset(self, rlds_config):
        return make_interleaved_dataset(**rlds_config)

    def __iter__(self) -> Dict[str, Any]:
        for rlds_batch in self.dataset.as_numpy_iterator():
            yield rlds_batch

    def __len__(self) -> int:
        return self.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")

