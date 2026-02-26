"""Load multiple checkpoints, merge their weights, and serve as a single policy.

Example usage:
    CUDA_VISIBLE_DEVICES=4,5 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/merging_experiments.py \
        --port 8100 \
        --config pi0_fast_droid \
        --merging_fn linear_interpolation \
        --merging_fn_kwargs '{"model_mixing_coefficients": [0.5, 0.5]}' \
        --checkpoint_dirs /path/to/finetuned/checkpoint  gs://openpi-assets/checkpoints/pi0_fast_droid
"""

import ast
import dataclasses
import logging
import socket
from typing import Any

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


@dataclasses.dataclass
class Args:
    """Arguments for the merged policy server."""

    port: int
    config: str
    merging_fn: str
    merging_fn_kwargs: str
    checkpoint_dirs: list[str]
    default_prompt: str | None = None
    record: bool = False

    def parsed_kwargs(self) -> dict[str, Any]:
        if self.merging_fn_kwargs is None:
            return {}
        return ast.literal_eval(self.merging_fn_kwargs)


def main(args: Args) -> None:
    args.merging_fn_kwargs = args.parsed_kwargs()
    policy = _policy_config.create_merged_policy(
        _config.get_config(args.config),
        args.checkpoint_dirs,
        args.merging_fn,
        args.merging_fn_kwargs,
        default_prompt=args.default_prompt,
    )
    policy_metadata = policy.metadata

    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
