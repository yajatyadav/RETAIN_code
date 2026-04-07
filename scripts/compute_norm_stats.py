"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as lerobot_data_loader
import openpi.training.rlds_dataloader.dataloader as rlds_data_loader
import openpi.transforms as transforms
import openpi.training.sharding as sharding
import jax

class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def main(config_name: str, max_batches: int | None = None, batch_size: int = 128):
    config = _config.get_config(config_name)
    if config.is_RLDS:
        data_loader = rlds_data_loader.create_data_loader(
            config, skip_norm_stats=True
        )
    else:
        data_loader = lerobot_data_loader.create_data_loader(
            config, skip_norm_stats=True
        )
    
    if max_batches is None:
        max_transitions = 52042 + 66984 + 52970 + 567494 # TODO(YY): hardcoding libero-pretraining dataset size for now...
        num_batches = max_transitions // batch_size
    else:
        num_batches = max_batches

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys} 

    for i, batch in tqdm.tqdm(enumerate(data_loader), total=num_batches, desc="Computing stats"):
        if i >= num_batches:
            break
        obs, action = batch
        obs = _model.Observation.to_dict(obs)
        action = action[:, 0, :] # the dataloader API returns chunked-actions, so just take the first action
        batch = {**obs, "actions": action}
        for key in keys:
            # if i == 0:
                # print("Batch keys: ", batch.keys())
                # print("Batch state shape + dtype: ", batch["state"].shape, batch["state"].dtype)
                # print("Batch actions shape + dtype: ", batch["actions"].shape, batch["actions"].dtype)
                # print("sample state: ", batch["state"][0])
                # print("sample actions: ", batch["actions"][0])
            values = np.asarray(batch[key])
            stats[key].update(values.reshape(-1, values.shape[-1]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}
    data_config = config.data.create(config.assets_dirs, config.model)
    output_path = config.assets_dirs / data_config.repo_id
    print("NORM_STATS ARE : ", norm_stats)
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)