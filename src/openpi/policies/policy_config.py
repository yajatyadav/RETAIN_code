import logging
import pathlib
from typing import Any, Callable

import jax
import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms

from openpi.policies.model_merging import merging_functions


def create_merged_policy(
    train_config: _config.TrainConfig,
    checkpoint_dirs: list[pathlib.Path | str],
    merging_fn: str,
    merging_fn_kwargs: dict[str, Any] | None = None,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
) -> _policy.Policy:
    """Create a policy from a merged checkpoint.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_name_strs = ["_".join(str(dir).split("/")[-2:]) for dir in checkpoint_dirs]
    checkpoint_dirs = [download.maybe_download(str(dir)) for dir in checkpoint_dirs]
    chekpoint_dir_to_load_norm_stats_from = checkpoint_dirs[0] ## they should all have the same norm stats, so just use the first one to load them

    print(f"Available merging functions: {merging_functions.keys()}, and will be using {merging_fn}")
    merging_fn = merging_functions.get(merging_fn)
    ## responsible for merging AND loading params into a model
    model = merging_fn(train_config, checkpoint_dirs, merging_fn_kwargs)

    ## rest is identical to create_trained_policy
    if train_config.data_list is not None:
        data_config = train_config.data_list[0].create(train_config.assets_dirs, train_config.model)
    else:
        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(chekpoint_dir_to_load_norm_stats_from / "assets", data_config.asset_id)

    
    # dir_name = f"MERGING_{merging_fn}_{'___'.join(checkpoint_name_strs)}"
    # debug_dir = f"/home/yajatyadav/generalist_finetuning/openpi_finetune/debug_inference_outputs/{dir_name}"
    suffix = ""
    for k, v in merging_fn_kwargs.items():
        suffix += f"_{k}_{v}"
    # debug_dir = f"/home/yajatyadav/generalist_FT_reversions/libero_eval_data/DEBUG_{merging_fn}___{suffix}"
    debug_dir = "/global/scratch/users/yajatyadav/research/debug_inference_outputs"
    
    return _policy.Policy(
        model,
        debug_dir=debug_dir,
        model_type=train_config.model.model_type,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
    )



    
    

def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir_str = str(checkpoint_dir)
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    logging.info("Loading model...")
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))

    ## YY: hack for cotraining
    if train_config.data_list is not None:
        data_config = train_config.data_list[0].create(train_config.assets_dirs, train_config.model)
    else:
        data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    exp_name, ckpt_number = checkpoint_dir_str.split("/")[-2:]
    dir_name = f"{exp_name}__ckpt_{ckpt_number}"
    # debug_dir = f"/home/yajatyadav/generalist_finetuning/openpi_finetune/debug_inference_outputs/{dir_name}"
    debug_dir = "/global/scratch/users/yajatyadav/research/debug_inference_outputs"
    
    return _policy.Policy(
        model,
        debug_dir=debug_dir,
        model_type=train_config.model.model_type,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
    )
