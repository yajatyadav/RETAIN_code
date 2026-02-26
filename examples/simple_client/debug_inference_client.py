import dataclasses
import enum
import logging
import time

import numpy as np
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro
from openpi.training.rlds_dataloader.dataloader import create_dataset
from openpi.training.config import get_config
import torch
from tqdm import tqdm

##YY: use this file to serve as a dummy client to debug the inference server!

"""
Usage:
uv run debug_inference_client.py --train_config <config> --port 8080
"""

class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Args:
    train_config: str
    host: str = "0.0.0.0"
    port: int = 8000

    env: EnvMode = EnvMode.DROID
    num_steps: int = 100
    step_size: int = 10

## YY: for debugging, train_dataset returns tensors, which we convert to numpy so we can serialize and "send over the network"
def convert_tensors_to_numpy(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            if v.dim() == 0:
                v = v.unsqueeze(0)
            new_val = v.detach().cpu().numpy()  # convert to numpy
            out[k] = new_val
        # elif isinstance(v, dict):
        #     out[k] = convert_tensors_to_numpy(v)  # recurse if nested dict
        else:
            out[k] = v
    return out

## YY: this function is now being used to load in the train_dataset, and query the policy server using these images. The server now also returns the computed loss in this case (since we are sending in the ground truth action)
## and these returned losses match the converged losses from the training loop!!
def main(args: Args) -> None:
    policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logging.info(f"Server metadata: {policy.get_server_metadata()}")
    config = get_config(args.train_config)
    
    if config.data_list is not None:
        data_config = config.data_list[0].create(config.assets_dirs, config.model)
    else:
        data_config = config.data.create(config.assets_dirs, config.model)
    train_dataset = create_dataset(config, config.model)
    repack_transform = data_config.repack_transforms.inputs[0]

    ## on the server side, DataTransforms and ModelTransforms are applied to the observation before inference, thus we will manually only apply the RepackTransform here
    ## we also send in the ground action so the server can compute the loss and return it back to us, (in actual inference, this would never happen!!)
    losses = []
    start = time.time()
    count = 1
    for i, train_sample in tqdm(enumerate(train_dataset), desc="DEBUG: Running inference on training dataset", total=args.num_steps * args.step_size):
        if i % args.step_size != 0:
            continue
        if count == args.num_steps:
            break
        
        count += 1
        sample_train_obs = repack_transform(convert_tensors_to_numpy(train_sample))
        server_response = policy.infer(sample_train_obs)
        if "loss" in server_response:
            loss = server_response["loss"]
            losses.append(loss)
            print(f"DEBUG: Loss: {loss:.4f}")
        if "actions" in server_response:
            actions = server_response["actions"]
            print("DEBUG: Actions: ", actions)
        if "gt_actions" in server_response:
            gt_actions = server_response["gt_actions"]
            print("DEBUG: GT Actions: ", gt_actions)
    
    avg_loss = sum(losses) / len(losses)
    end = time.time()
    
    print(f"Total time taken: {end - start:.2f} s")
    print(f"Average inference time: {1000 * (end - start) / args.num_steps:.2f} ms")
    print(f"Average Loss over {args.num_steps} samples drawn from Training Dataset: {avg_loss}")


def _random_observation_aloha() -> dict:
    return {
        "state": np.ones((14,)),
        "images": {
            "cam_high": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_low": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }



def _random_observation_droid() -> dict:
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def _random_observation_libero() -> dict:
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
