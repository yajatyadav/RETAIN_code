from collections.abc import Iterator, Sequence
import multiprocessing
import os
import typing
from typing import Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch.utils.data import IterableDataset

import openpi.models.model as _model
import openpi.training.config as _config
import openpi.transforms as _transforms

from openpi.training.rlds_dataloader.rlds_dataset import RLDSDataset

T_co = TypeVar("T_co", covariant=True)

## TODO(YY): still need to add: corruption check, idlestep filtering, propating the seed, sharding, mp_context, worker_init_fn, AND SO ON

class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


# class TransformedDataset(Dataset[T_co]):
#     def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
#         self._dataset = dataset
#         self._transform = _transforms.compose(transforms)

#     def __getitem__(self, index: SupportsIndex) -> T_co:
#         return self._transform(self._dataset[index])

#     def __len__(self) -> int:
#         return len(self._dataset)
    

class TransformedRLDSDataset(IterableDataset):
    def __init__(self, dataset: RLDSDataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
    
    def __iter__(self):
        for sample in self._dataset:
            yield self._transform(sample)

    def __len__(self) -> int:
        return self._dataset.dataset_length

    # === Explicitly Unused ===
    def __getitem__(self, idx: int) -> None:
        raise NotImplementedError("IterableDataset does not implement map-style __getitem__; see __iter__ instead!")
    

## YY: note that the returned dataset performs sampling WITHOUT replacement on the dataset_list. the iterator will exhaust as soon as a dataset is exhausted
def create_dataset(config: _config.TrainConfig, model_config: _model.BaseModelConfig) -> Dataset:
    """Create a dataset for training."""
    if config.data_list is not None:
        data_configs = config.data_list
        data_mix_weights = config.data_mix_weights
    else:
        data_configs = [config.data]
        data_mix_weights = [1.0]
    
    data_configs = [data_config.create(config.assets_dirs, config.model) for data_config in data_configs]
    repo_ids = [data_config.repo_id for data_config in data_configs]
    mixture_spec = [(repo_id, weight) for repo_id, weight in zip(repo_ids, data_mix_weights)]
    action_key_list = [data_config.action_sequence_keys[0] for data_config in data_configs]
    return RLDSDataset(
        data_root_dir = config.data_root_dir,
        mixture_spec = mixture_spec,
        action_key_list = action_key_list, ## TODO(YY): this is a hacky way to propagate the action key to the RLDS logic, need to fix this
        batch_size=config.batch_size // jax.process_count(),
        num_workers=config.num_workers,
        action_horizon=model_config.action_horizon,
        shuffle_buffer_size=100_000,
    )  
    # dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    # episodes_with_all_videos = []
    # if data_config.episodes_with_all_videos_path is None:
    #     episodes_with_all_videos = list(range(len(dataset_meta.episodes)))
    # else:
    #     episodes_with_all_videos = np.load(data_config.episodes_with_all_videos_path).tolist()
    #     print(f"✅✅✅✅ Create_Dataset is ignoring episodes without mp4 recordings, there are {len(dataset_meta.episodes) - len(episodes_with_all_videos)} episodes without mp4, out of a total of {len(dataset_meta.episodes)} episodes considered ✅✅✅✅")
    # dataset = lerobot_dataset.LeRobotDataset(
    #     data_config.repo_id,
    #     episodes=episodes_with_all_videos,
    #     delta_timestamps={
    #         key: [t / dataset_meta.fps for t in range(model_config.action_horizon)]
    #         for key in data_config.action_sequence_keys
    #     },
    # )

    # if data_config.prompt_from_task:
    #     dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset_meta.tasks)])


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return TransformedRLDSDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
    )

class RLDSDataLoader():
    def __init__(self, config: _config.TrainConfig, data_config: _config.DataConfig, dataset: Dataset, sharding: jax.sharding.Sharding | None = None, num_batches: int | None = None):
            self._data_config = data_config
            if sharding is None:
                sharding = jax.sharding.NamedSharding(
                    jax.sharding.Mesh(jax.devices(), ("B",)),
                    jax.sharding.PartitionSpec("B"),
                    )   
            self._sharding = sharding
            self._num_batches = num_batches
            num_workers = 0 # causes conflicts with tfds's own multiprocessing logic...
            generator = torch.Generator()
            generator.manual_seed(config.seed)
            self._data_loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=config.batch_size // jax.process_count(),
                shuffle=False, # not allowing shuffling here, since TFDS will handle it!!
                num_workers=num_workers,
                multiprocessing_context=multiprocessing.get_context("spawn") if num_workers > 0 else None,
                persistent_workers=num_workers > 0,
                collate_fn=_collate_fn,
                worker_init_fn=_worker_init_fn,
                drop_last=True,
                generator=generator,
            )
    
    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            print(f" 🧟‍♂️🧟‍♂️🧟‍♂️🧟‍♂️ Dataloader Exhuasted/First Epoch: creating a new iterator over the dataloader. 🧟‍♂️🧟‍♂️🧟‍♂️🧟‍♂️")
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)



## TODO(YY): since we are throwing away num_workers here, make sure we correctly pass it onto the TFDS code!!!
def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
    """
    ## YY: in the cotraining case, it's fine to just use the first data's config, as all data sources MUST use the same norm_stats, repack_transforms, data_transforms, model_transforms, etc. anyways
    ## it does make DataLoaderImpl.data_config() wrong, but it's not really used anywhere other than some checkpointing logic :)
    if config.data_list is not None:
        data_config = config.data_list[0].create(config.assets_dirs, config.model)
    else:
        data_config = config.data.create(config.assets_dirs, config.model)
        

    dataset = create_dataset(config, config.model)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)
    dataloader = RLDSDataLoader(config, data_config, dataset, sharding, num_batches)


    class DataLoaderImpl(DataLoader):
        def __init__(self, data_config: _config.DataConfig, dataloader: RLDSDataLoader):
            self._data_config = data_config
            self._dataloader = dataloader
        
        def data_config(self) -> _config.DataConfig:
            return self._data_config
        
        def __iter__(self):
            for batch in self._dataloader:
                yield _model.Observation.from_dict(batch), batch["actions"]      
    
    return DataLoaderImpl(data_config, dataloader)

    # data_loader = TorchDataLoader(
    #     dataset,
    #     local_batch_size=config.batch_size // jax.process_count(),
    #     sharding=sharding,
    #     shuffle=shuffle,
    #     num_batches=num_batches,
    #     num_workers=num_workers,
    #     seed=config.seed,
    # )

    # class DataLoaderImpl(DataLoader):
    #     def __init__(self, data_config: _config.DataConfig, dataset: Dataset):
    #         self._data_config = data_config
    #         local_batch_size = config.batch_size // jax.process_count()          

    #         ## YY: explicitly disable, since TFDS handle multiprocessing internally
    #         mp_context = None
    #         num_workers = 0
            
    #         generator = torch.Generator()
    #         generator.manual_seed(config.seed)

    #         self._data_loader = torch.utils.data.DataLoader(
    #         typing.cast(torch.utils.data.Dataset, dataset),
    #         batch_size=local_batch_size,
    #         shuffle=shuffle,
    #         num_workers=num_workers,
    #         multiprocessing_context=mp_context,
    #         persistent_workers=num_workers > 0,
    #         collate_fn=_collate_fn,
    #         worker_init_fn=_worker_init_fn,
    #         drop_last=True,
    #         generator=generator,
    #     )

    #     def data_config(self) -> _config.DataConfig:
    #         return self._data_config

    #     def __iter__(self):
    #         for batch in self._data_loader:
    #             yield _model.Observation.from_dict(batch), batch["actions"]

    # return DataLoaderImpl(data_config, dataset)


# class TorchDataLoader:
#     def __init__(
#         self,
#         dataset,
#         local_batch_size: int,
#         *,
#         sharding: jax.sharding.Sharding | None = None,
#         shuffle: bool = False,
#         num_batches: int | None = None,
#         num_workers: int = 0,
#         seed: int = 0,
#     ):
#         """Create a PyTorch data loader.

#         Args:
#             dataset: The dataset to load.
#             local_batch_size: The local batch size for each process.
#             sharding: The sharding to use for the data loader.
#             shuffle: Whether to shuffle the data.
#             num_batches: If provided, determines the number of returned batches. If the
#                 number is larger than the number of batches in the dataset, the data loader
#                 will loop over the dataset. If not provided, will iterate over the dataset
#                 indefinitely.
#             num_workers: The number of worker processes to use. If zero, the data loader will
#                 execute in the main process.
#             seed: The seed to use for shuffling the data.
#         """
#         if jax.process_count() > 1:
#             raise NotImplementedError("Data loading with multiple processes is not supported.")

#         if len(dataset) < local_batch_size:
#             raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

#         if sharding is None:
#             # Use data parallel sharding by default.
#             sharding = jax.sharding.NamedSharding(
#                 jax.sharding.Mesh(jax.devices(), ("B",)),
#                 jax.sharding.PartitionSpec("B"),
#             )

#         self._sharding = sharding
#         self._num_batches = num_batches

#         mp_context = None
#         if num_workers > 0:
#             mp_context = multiprocessing.get_context("spawn")

#         generator = torch.Generator()
#         generator.manual_seed(seed)
#         print(f"🥰🥰🥰 Creating base torch dataloader with num_workers={num_workers}, local_batch_size={local_batch_size}, shuffle={shuffle}, multiprocessing_context={mp_context}, persistent_workers={num_workers > 0} 🥰🥰🥰")
#         self._data_loader = torch.utils.data.DataLoader(
#             typing.cast(torch.utils.data.Dataset, dataset),
#             batch_size=local_batch_size,
#             shuffle=shuffle,
#             num_workers=num_workers,
#             multiprocessing_context=mp_context,
#             persistent_workers=num_workers > 0,
#             collate_fn=_collate_fn,
#             worker_init_fn=_worker_init_fn,
#             drop_last=True,
#             generator=generator,
#         )

#     @property
#     def torch_loader(self) -> torch.utils.data.DataLoader:
#         return self._data_loader

#     def __iter__(self):
#         num_items = 0
#         while True:
#             data_iter = iter(self._data_loader)
#             while True:
#                 if self._num_batches is not None and num_items >= self._num_batches:
#                     return
#                 try:
#                     batch = next(data_iter)
#                 except StopIteration:
#                     break  # We've exhausted the dataset. Create a new iterator and start over.
#                 num_items += 1
#                 yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *x: np.stack(np.asarray(x), axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"