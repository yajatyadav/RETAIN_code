#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, UP017
"""读取真实 RLDS 单批数据，验证 RETAIN 训练输入的完整 transform 链。"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import jax
import numpy as np

from openpi.training import config as _config
from openpi.training.rlds_dataloader import dataloader as _dataloader

DEFAULT_CONFIGS = (
    "retain_repro_pretrain",
    "retain_repro_task_ft_stove",
    "retain_repro_task_ft_mugs",
    "retain_repro_task_ft_basket",
    "retain_repro_coft_stove",
    "retain_repro_coft_mugs",
    "retain_repro_coft_basket",
)


def atomic_json_dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def inspect_config(name: str) -> dict:
    config = _config.get_config(name)
    loader = _dataloader.create_data_loader(
        config,
        num_workers=0,
        shuffle=False,
        num_batches=1,
    )
    observation, actions = next(iter(loader))
    state = np.asarray(jax.device_get(observation.state))
    actions_array = np.asarray(jax.device_get(actions))
    images = {
        key: list(np.asarray(jax.device_get(value)).shape)
        for key, value in observation.images.items()
    }
    result = {
        "config": name,
        "batch_size": config.batch_size,
        "data_mix_weights": config.data_mix_weights,
        "jax_devices": [str(device) for device in jax.devices()],
        "state_shape": list(state.shape),
        "state_dtype": str(state.dtype),
        "state_finite": bool(np.isfinite(state).all()),
        "action_shape": list(actions_array.shape),
        "action_dtype": str(actions_array.dtype),
        "action_finite": bool(np.isfinite(actions_array).all()),
        "image_shapes": images,
        "prompt_tokens_shape": list(observation.tokenized_prompt.shape),
    }
    expected_state = (config.batch_size, config.model.action_dim)
    expected_actions = (
        config.batch_size,
        config.model.action_horizon,
        config.model.action_dim,
    )
    if state.shape != expected_state:
        raise RuntimeError(f"{name} state shape {state.shape} != {expected_state}")
    if actions_array.shape != expected_actions:
        raise RuntimeError(f"{name} action shape {actions_array.shape} != {expected_actions}")
    if not result["state_finite"] or not result["action_finite"]:
        raise RuntimeError(f"{name} 出现 NaN/Inf")
    print(f"训练输入 smoke test 通过: {name}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="*", default=list(DEFAULT_CONFIGS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = {
        "generated_at": dt.datetime.now(dt.timezone.utc).astimezone().isoformat(),
        "configs": [inspect_config(name) for name in args.configs],
    }
    atomic_json_dump(args.output.resolve(), payload)
    print(f"全部输入 smoke tests 通过，记录: {args.output.resolve()}")


if __name__ == "__main__":
    main()
