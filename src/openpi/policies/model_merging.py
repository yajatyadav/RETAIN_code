"""Model merging functions for combining multiple VLA checkpoints.
This files provides many model merging logic, though all experiemnts in the paper (https://arxiv.org/abs/2512.08333)
are conducted using linear interpolation.

Each merging function accepts a TrainConfig, a list of checkpoint directories,
and optional kwargs, and returns a merged model ready for inference.

Convention: the *last* entry in ``checkpoint_dirs`` is treated as the pretrained
base model; all preceding entries are finetuned checkpoints.
"""

import logging
import pathlib
import re
from typing import Any

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp

import openpi.models.model as _model
import openpi.shared.array_typing as at
from openpi.training import config as _config


def validate_merging_coefficients(coefficients: list[float]) -> None:
    assert all(0 <= w <= 1 for w in coefficients), "Mixing coefficients must be in [0, 1]."
    assert abs(sum(coefficients) - 1.0) < 1e-6, "Mixing coefficients must sum to 1."


def validate_params_list(
    train_config: _config.TrainConfig, params_list: list[at.Params], check_dtypes: bool = False
) -> None:
    model = nnx.eval_shape(train_config.model.create, jax.random.key(0))
    state = nnx.state(model)
    for params in params_list:
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=check_dtypes)


def linearly_merge_params(params_list: list[at.Params], mixing_coefficients: list[float]) -> at.Params:
    """Compute a weighted average of parameter pytrees."""
    logging.info(f"Merging {len(params_list)} param sets with coefficients {mixing_coefficients}")
    trees = [jax.tree.map(lambda x: x * w, p) for w, p in zip(mixing_coefficients, params_list)]
    return jax.tree.map(lambda *x: jnp.sum(jnp.stack(x), axis=0), *trees)


# ---------------------------------------------------------------------------
# Linear interpolation
# ---------------------------------------------------------------------------

def linear_interpolation(
    train_config: _config.TrainConfig,
    checkpoint_dirs: list[pathlib.Path | str],
    merging_fn_kwargs: dict[str, Any] | None = None,
) -> at.Params:
    """Weighted linear average of all checkpoint parameters."""
    model_mixing_coefficients = merging_fn_kwargs["model_mixing_coefficients"]
    validate_merging_coefficients(model_mixing_coefficients)
    logging.info(f"Linear interpolation of {len(checkpoint_dirs)} checkpoints with weights {model_mixing_coefficients}")

    params_list = [_model.restore_params(d / "params", dtype=jnp.bfloat16) for d in checkpoint_dirs]
    validate_params_list(train_config, params_list)

    merged_params = linearly_merge_params(params_list, model_mixing_coefficients)
    return train_config.model.load(merged_params)


# ---------------------------------------------------------------------------
# Spherical linear interpolation (SLERP)
# ---------------------------------------------------------------------------

def _slerp_merge_params(
    params_list: list[at.Params], t: float, dot_threshold: float, eps: float
) -> at.Params:
    """SLERP between two parameter pytrees. Only supports exactly 2 checkpoints."""
    base_params, ft_params = params_list

    def normalize(arr):
        norm = jnp.linalg.norm(arr)
        return jnp.where(norm > eps, arr / norm, arr)

    def slerp_fn(base_arr, ft_arr):
        base_normed = normalize(base_arr)
        ft_normed = normalize(ft_arr)
        dot = jnp.clip(jnp.sum(base_normed * ft_normed), -1, 1)
        # Fall back to linear interpolation for nearly collinear vectors.
        if jnp.abs(dot) > dot_threshold:
            return (1 - t) * base_arr + t * ft_arr
        theta_0 = jnp.arccos(dot)
        sin_theta_0 = jnp.sin(theta_0)
        theta_t = theta_0 * t
        s0 = jnp.sin(theta_0 - theta_t) / sin_theta_0
        s1 = jnp.sin(theta_t) / sin_theta_0
        return s0 * base_arr + s1 * ft_arr

    return jax.tree.map(slerp_fn, base_params, ft_params)


def spherical_linear_interpolation(
    train_config: _config.TrainConfig,
    checkpoint_dirs: list[pathlib.Path | str],
    merging_fn_kwargs: dict[str, Any] | None = None,
    dot_threshold: float = 0.9995,
    eps: float = 1e-8,
) -> at.Params:
    """SLERP between two checkpoints. ``t`` in [0, 1]: higher favours finetuned."""
    assert len(checkpoint_dirs) == 2, "SLERP only supports exactly 2 checkpoints."
    base_params = _model.restore_params(checkpoint_dirs[-1] / "params", dtype=jnp.bfloat16)
    ft_params = _model.restore_params(checkpoint_dirs[0] / "params", dtype=jnp.bfloat16)
    params_list = [base_params, ft_params]
    validate_params_list(train_config, params_list)

    dot_threshold = merging_fn_kwargs.get("DOT_THRESHOLD", dot_threshold)
    eps = merging_fn_kwargs.get("eps", eps)
    t = merging_fn_kwargs["t"]
    merged = _slerp_merge_params(params_list, t, dot_threshold, eps)
    return train_config.model.load(merged)


# ---------------------------------------------------------------------------
# Per-module (multi-modal) linear interpolation
# ---------------------------------------------------------------------------

def _filter_keys(flat_param, include_patterns=None, exclude_patterns=None):
    """Return subset of a flattened param dict whose keys match the given patterns.

    Args:
        flat_param: Dict with tuple keys (from ``traverse_util.flatten_dict``).
        include_patterns: Regex patterns that must *all* match.
        exclude_patterns: Regex patterns that must *not* match.
    """
    include_patterns = include_patterns or []
    exclude_patterns = exclude_patterns or []
    filtered = {}
    for key, value in flat_param.items():
        key_str = "/".join(key)
        if all(re.search(p, key_str) for p in include_patterns):
            if not any(re.search(p, key_str) for p in exclude_patterns):
                filtered[key] = value
    return filtered


def multimodal_linear_interpolation(
    train_config: _config.TrainConfig,
    checkpoint_dirs: list[pathlib.Path | str],
    merging_fn_kwargs: dict[str, Any] | None = None,
) -> at.Params:
    """Per-module linear interpolation for pi0-family models.

    Merges vision encoder, LLM backbone, and action expert parameters with
    independent mixing coefficients.  Auxiliary action projections (in/out proj,
    time MLPs, state proj) use the action expert coefficients.
    """
    vision_coeffs = merging_fn_kwargs["vision_mixing_coefficients"]
    llm_coeffs = merging_fn_kwargs["llm_mixing_coefficients"]
    action_coeffs = merging_fn_kwargs["action_expert_mixing_coefficients"]
    validate_merging_coefficients(vision_coeffs)
    validate_merging_coefficients(llm_coeffs)
    validate_merging_coefficients(action_coeffs)

    params_list = [_model.restore_params(d / "params", dtype=jnp.bfloat16) for d in checkpoint_dirs]
    validate_params_list(train_config, params_list)

    model = nnx.eval_shape(train_config.model.create, jax.random.key(0))
    graphdef, state = nnx.split(model)
    for params in params_list:
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)

    flat_params_list = [traverse_util.flatten_dict(p) for p in params_list]

    # Vision params live under "img"; LLM backbone under "llm" (excluding "_1");
    # action expert under "llm" *with* "_1".
    vision_merged = linearly_merge_params(
        [_filter_keys(fp, include_patterns=[r"img"]) for fp in flat_params_list], vision_coeffs
    )
    llm_merged = linearly_merge_params(
        [_filter_keys(fp, include_patterns=[r"llm"], exclude_patterns=[r"_1"]) for fp in flat_params_list], llm_coeffs
    )
    action_expert_merged = linearly_merge_params(
        [_filter_keys(fp, include_patterns=[r"llm", r"_1"]) for fp in flat_params_list], action_coeffs
    )

    # Auxiliary action-related projections use the action expert coefficients.
    aux_patterns = [r"action_in_proj", r"action_out_proj", r"action_time_mlp_in", r"action_time_mlp_out", r"state_proj"]
    aux_merged = {}
    for pattern in aux_patterns:
        merged = linearly_merge_params(
            [_filter_keys(fp, include_patterns=[pattern]) for fp in flat_params_list], action_coeffs
        )
        aux_merged.update(merged)

    new_params = {**vision_merged, **llm_merged, **action_expert_merged, **aux_merged}
    new_partial_params = traverse_util.unflatten_dict(new_params)

    state.replace_by_pure_dict(new_partial_params)
    model = nnx.merge(graphdef, state)
    return model


# ---------------------------------------------------------------------------
# Task vector interpolation (continual learning)
# ---------------------------------------------------------------------------

def _make_task_vector(base_params: at.Params, ft_params: at.Params) -> at.Params:
    return jax.tree.map(lambda ft, base: ft - base, ft_params, base_params)


def task_vector_interpolation(
    train_config: _config.TrainConfig,
    checkpoint_dirs: list[pathlib.Path | str],
    merging_fn_kwargs: dict[str, Any] | None = None,
) -> at.Params:
    """Additive task-vector merging for continual learning.

    Expects N finetuned checkpoint dirs followed by 1 base checkpoint dir.
    ``lambda_list`` controls the scaling of each task vector.
    """
    assert len(checkpoint_dirs) > 2, "Need at least 2 finetuned models + 1 base model."
    lambda_list = merging_fn_kwargs["lambda_list"]
    assert len(checkpoint_dirs) == len(lambda_list) + 1, "len(checkpoint_dirs) must equal len(lambda_list) + 1"
    assert all(0 <= l <= 1 for l in lambda_list), "Lambda values must be in [0, 1]."

    logging.info(f"Loading base params from {checkpoint_dirs[-1]}")
    base_params = _model.restore_params(checkpoint_dirs[-1] / "params", dtype=jnp.bfloat16)

    task_vectors = []
    for i, checkpoint_dir in enumerate(checkpoint_dirs[:-1]):
        logging.info(f"Building task vector for {checkpoint_dir} with lambda={lambda_list[i]}")
        ft_params = _model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16)
        tv = _make_task_vector(base_params, ft_params)
        tv = jax.tree.map(lambda x: x * lambda_list[i], tv)
        task_vectors.append(tv)

    final_tv = jax.tree.map(lambda *x: jnp.sum(jnp.stack(x), axis=0), *task_vectors)
    logging.info("Merging task vectors into base params")
    merged_params = jax.tree.map(lambda base, task: base + task, base_params, final_tv)
    return train_config.model.load(merged_params)


# ---------------------------------------------------------------------------
# DARE (Drop And REscale) interpolation
# ---------------------------------------------------------------------------

def _make_random_mask_pytree(rng, tree, dropout_prob):
    """Create a Bernoulli(1 - dropout_prob) mask matching the pytree structure."""
    leaves, treedef = jax.tree.flatten(tree)
    leaf_rngs = jax.random.split(rng, len(leaves))
    mask_leaves = [jax.random.bernoulli(r, 1 - dropout_prob, shape=l.shape) for r, l in zip(leaf_rngs, leaves)]
    return jax.tree.unflatten(treedef, mask_leaves)


def _dare_core(base_params, ft_params, dropout_prob, seed):
    """Shared DARE logic: compute masked and rescaled task vector."""
    task_vector = _make_task_vector(base_params, ft_params)
    key = jax.random.key(seed)
    mask = _make_random_mask_pytree(key, base_params, dropout_prob)
    masked_tv = jax.tree.map(lambda tv, m: tv * m, task_vector, mask)
    rescaled_tv = jax.tree.map(lambda x: x / (1 - dropout_prob), masked_tv)

    leaf_counts = jax.tree_util.tree_map(lambda x: x.size, rescaled_tv)
    total_elements = jax.tree_util.tree_reduce(lambda a, b: a + b, leaf_counts)
    leaf_zeros = jax.tree_util.tree_map(lambda x: jnp.sum(x == 0, dtype=jnp.int64), rescaled_tv)
    total_zeros = jax.tree_util.tree_reduce(lambda a, b: a + b, leaf_zeros)
    logging.info(f"DARE: {total_zeros}/{total_elements} elements zeroed (dropout_prob={dropout_prob})")

    return rescaled_tv


def dare_interpolation(
    train_config: _config.TrainConfig,
    checkpoint_dirs: list[pathlib.Path | str],
    merging_fn_kwargs: dict[str, Any] | None = None,
) -> at.Params:
    """DARE: randomly drop task-vector entries, rescale, then add to base."""
    assert len(checkpoint_dirs) == 2, "DARE only supports exactly 2 checkpoints."
    base_params = _model.restore_params(checkpoint_dirs[-1] / "params", dtype=jnp.bfloat16)
    ft_params = _model.restore_params(checkpoint_dirs[0] / "params", dtype=jnp.bfloat16)

    rescaled_tv = _dare_core(base_params, ft_params, merging_fn_kwargs["dropout_prob"], merging_fn_kwargs["seed"])

    scale = merging_fn_kwargs["task_vector_scaling_factor"]
    merged = jax.tree.map(lambda base, tv: base + scale * tv, base_params, rescaled_tv)
    at.check_pytree_equality(expected=base_params, got=merged, check_shapes=True, check_dtypes=True)
    return train_config.model.load(merged)


def dare_slerp_interpolation(
    train_config: _config.TrainConfig,
    checkpoint_dirs: list[pathlib.Path | str],
    merging_fn_kwargs: dict[str, Any] | None = None,
) -> at.Params:
    """DARE + SLERP: apply DARE to the task vector, then SLERP between base and result."""
    assert len(checkpoint_dirs) == 2, "DARE-SLERP only supports exactly 2 checkpoints."
    base_params = _model.restore_params(checkpoint_dirs[-1] / "params", dtype=jnp.bfloat16)
    ft_params = _model.restore_params(checkpoint_dirs[0] / "params", dtype=jnp.bfloat16)

    rescaled_tv = _dare_core(base_params, ft_params, merging_fn_kwargs["dropout_prob"], merging_fn_kwargs["seed"])
    new_ft_params = jax.tree.map(lambda base, tv: base + tv, base_params, rescaled_tv)

    t = merging_fn_kwargs["t"]
    dot_threshold = merging_fn_kwargs.get("DOT_THRESHOLD", 0.9995)
    eps = merging_fn_kwargs.get("eps", 1e-8)
    merged = _slerp_merge_params([base_params, new_ft_params], t, dot_threshold, eps)
    return train_config.model.load(merged)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

merging_functions = {
    "linear_interpolation": linear_interpolation,
    "slerp_interpolation": spherical_linear_interpolation,
    "multimodal_linear_interpolation": multimodal_linear_interpolation,
    "dare_interpolation": dare_interpolation,
    "dare_slerp_interpolation": dare_slerp_interpolation,
    "task_vector_interpolation": task_vector_interpolation,
}
