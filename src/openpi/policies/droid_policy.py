import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
import torch

def make_droid_example() -> dict:
    """Creates a random input example for the Droid policy."""
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image



# def _parse_image_rlds_batched(image_batch) -> np.ndarray:
#     image_batch = np.asarray(image_batch)
#     # Remove the singleton dimension (batch_size, 1, height, width, channels) -> (batch_size, height, width, channels)
#     image_batch = np.squeeze(image_batch, axis=1)
    
#     # Convert floating point to uint8 if needed
#     if np.issubdtype(image_batch.dtype, np.floating):
#         image_batch = (255 * image_batch).astype(np.uint8)
#     if image_batch.shape[1] == 3:
#         image_batch = einops.rearrange(image_batch, "b c h w -> b h w c")
#     return image_batch


def _parse_image_rlds(image) -> np.ndarray:
    image = np.asarray(image)
    # Remove the singleton dimension (batch_size, 1, height, width, channels) -> (batch_size, height, width, channels)
    image = np.squeeze(image)
    image = _parse_image(image)
    return image

@dataclasses.dataclass(frozen=True)
class DroidRLDSInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        ## YY: hack so that inference and training can both use this class
        if "state" in data:
            # RLDS training branch
            state = np.squeeze(data["state"])
            # state = transforms.pad_to_dim(state, self.action_dim)
        else: 
            # inference branch
            state = np.concatenate([data["observation/joint_position"], data["observation/gripper_position"]])
        
        base_image = _parse_image_rlds(data["observation/exterior_image_1_left"])
        wrist_image = _parse_image_rlds(data["observation/wrist_image_left"])
        
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            ## YY: need 2 branches for training/inference here as well
            prompt = data["prompt"]
            if isinstance(prompt, str):
                inputs["prompt"] = prompt
            else:
                inputs["prompt"] = prompt.decode("utf-8")
        return inputs



@dataclasses.dataclass(frozen=True)
class DroidInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        gripper_pos = np.asarray(data["observation/gripper_position"])
        if gripper_pos.ndim == 0:
            # Ensure gripper position is a 1D array, not a scalar, so we can concatenate with joint positions
            gripper_pos = gripper_pos[np.newaxis]
        state = np.concatenate([data["observation/joint_position"], gripper_pos])
        # state = transforms.pad_to_dim(state, self.action_dim) ## TODO(YY): no need for this anymore, but still confirm this is correct

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        base_image = _parse_image(data["observation/exterior_image_1_left"])
        wrist_image = _parse_image(data["observation/wrist_image_left"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.array(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DroidOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        return {"actions": np.asarray(data["actions"][:, :8])}


# @dataclasses.dataclass(frozen=True)
# ## TODO(YY): kind of scuffed, diff inputs class to get cadene/droid to work... when co-training, will need to reconcile this...
# class DroidInputsLeRobot(transforms.DataTransformFn):
#     # The action dimension of the model. Will be used to pad state and actions.
#     action_dim: int

#     # Determines which model will be used.
#     model_type: _model.ModelType = _model.ModelType.PI0

#     def __call__(self, data: dict) -> dict:
#         state = data["state"]
#         # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
#         # stores as float32 (C,H,W), gets skipped for policy inference
#         base_image = _parse_image(data["observation/exterior_image_1_left"])
#         wrist_image = _parse_image(data["observation/wrist_image_left"])

#         match self.model_type:
#             case _model.ModelType.PI0:
#                 names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
#                 images = (base_image, wrist_image, np.zeros_like(base_image))
#                 image_masks = (np.True_, np.True_, np.False_)
#             case _model.ModelType.PI0_FAST:
#                 names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
#                 # We don't mask out padding images for FAST models.
#                 images = (base_image, np.zeros_like(base_image), wrist_image)
#                 image_masks = (np.True_, np.True_, np.True_)
#             case _:
#                 raise ValueError(f"Unsupported model type: {self.model_type}")

#         inputs = {
#             "state": state,
#             "image": dict(zip(names, images, strict=True)),
#             "image_mask": dict(zip(names, image_masks, strict=True)),
#         }

#         if "actions" in data:
#             inputs["actions"] = transforms.pad_to_dim(data["actions"], self.action_dim) # TODO(YY): cadene/droid LeRobot only has 7D actions, unlike the 8D teleop data, so padding here
#         if "prompt" in data:
#             inputs["prompt"] = data["prompt"]
#         return inputs