import os
import uuid
import matplotlib.pyplot as plt

from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy



# def make_attn_mask(input_mask, mask_ar):
#     """Adapted from big_vision.

#     Tokens can attend to valid inputs tokens which have a cumulative mask_ar
#     smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
#     setup several types of attention, for example:

#       [[1 1 1 1 1 1]]: pure causal attention.

#       [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
#           themselves and the last 3 tokens have a causal attention. The first
#           entry could also be a 1 without changing behaviour.

#       [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
#           block can attend all previous blocks and all tokens on the same block.

#     Args:
#       input_mask: bool[B, N] true if its part of the input, false if padding.
#       mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
#         it and false where it shares the same attention mask as the previous token.
#     """
#     mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
#     cumsum = jnp.cumsum(mask_ar, axis=1)
#     attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
#     valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
#     return jnp.logical_and(attn_mask, valid_mask)

class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        debug_dir: str | None = None,
        *,
        model_type: _model.ModelType | None = None,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self.model_type = model_type

        ## TODO(YY): debug by not jitting the sample_actions function
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._compute_loss = model.compute_loss
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

        self._debug_dir = pathlib.Path(debug_dir)
        self._num_debug_outputs_saved = 0
        self._num_debug_outputs_to_save = 20
        os.makedirs(self._debug_dir, exist_ok=True)


    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        input_has_gt_action = obs.get("actions") is not None
        
        
        ## saving some debug outputs
        if self._num_debug_outputs_saved < self._num_debug_outputs_to_save:
            self._num_debug_outputs_saved += 1
            out_dir = self._debug_dir / f"call_{self._num_debug_outputs_saved}"
            os.makedirs(out_dir, exist_ok=True)
            
            obs_copy = jax.tree.map(lambda x: x, obs)
            obs_copy = self._input_transform(obs_copy)
            obs_copy = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], obs_copy)
            OBS = _model.Observation.from_dict(obs_copy)

            wrist_image_str  = "wrist_0_rgb" 
            if self.model_type in [_model.ModelType.PI05, _model.ModelType.PI0]:
                wrist_image_str = "left_wrist_0_rgb"
            sample_wrist_im = np.array(OBS.images[wrist_image_str][0])
            sample_wrist_im = (sample_wrist_im + 1)/2
            wrist_save_path = out_dir / "sample_wrist_im.png"
            plt.imsave(wrist_save_path, sample_wrist_im)
            
            sample_base_im = np.array(OBS.images["base_0_rgb"][0])
            sample_base_im = (sample_base_im + 1)/2          
            base_save_path = out_dir / "sample_base_im.png"
            plt.imsave(base_save_path, sample_base_im)
            
            sample_state = np.array(OBS.state[0])
            state_save_path = out_dir / "sample_state.txt"
            np.savetxt(state_save_path, sample_state, fmt="%.4f", delimiter=" ", header="Sample State Array", comments="# ")
            
            if input_has_gt_action:
                sample_actions = np.array(obs_copy["actions"][0])
                actions_save_path = out_dir / "sample_actions.txt"
                np.savetxt(actions_save_path, sample_actions, fmt="%.4f", delimiter=" ", header="Sample Actions Array", comments="# ")
            
            sample_tokenized_prompt = np.array(OBS.tokenized_prompt[0])
            tokenized_prompt_save_path = out_dir / "sample_tokenized_prompt.txt"
            np.savetxt(tokenized_prompt_save_path, sample_tokenized_prompt, fmt="%.4f", delimiter=" ", header="Sample Tokenized Prompt Array", comments="# ")



        ## YY: if the obs has the GT action, this is a debug run, so just compute loss and return. Also doesn't make sense to sample_actions
        ## as the tokenized_prompt already contains the action tokens in the postfix in this scenario (for the pi0fast case)

        if input_has_gt_action and self.model_type not in [_model.ModelType.PI05, _model.ModelType.PI0]:
            inputs_2 = jax.tree.map(lambda x: x, obs)
            inputs_2 = self._input_transform(inputs_2)
            inputs_2 = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs_2)
            # print("Client Provided Ground Truth Actions, computing loss against model Predictions")
            # print("CLIENT side actions shape: ", obs["actions"].shape)
            self._rng, sample_rng = jax.random.split(self._rng)
            chunked_loss = self._compute_loss(sample_rng, _model.Observation.from_dict(inputs_2), inputs_2["actions"], train=False)
            loss =jnp.mean(chunked_loss).item()
            # print("GT ACTIONS PROVIDED, Policy is returning the following outputs: ", {"loss": loss})
            return {"loss": loss}
        
        ## for pi0 and pi0.5, the actions are not in ground-truthobs's tokenized_prompt, so we can "call" sample_actions and return as if we are actually sampling actions!
        if input_has_gt_action and self.model_type in [_model.ModelType.PI05, _model.ModelType.PI0]:
            inputs_2 = jax.tree.map(lambda x: x, obs)
            inputs_2 = self._input_transform(inputs_2)
            inputs_2 = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs_2)
            
            self._rng, sample_rng = jax.random.split(self._rng)
            actions = self._sample_actions(sample_rng, _model.Observation.from_dict(inputs_2), **self._sample_kwargs)
            outputs = {"state": inputs_2["state"], "actions": actions}
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
            outputs = self._output_transform(outputs)
            
            self._rng, sample_rng = jax.random.split(self._rng)
            chunked_loss = self._compute_loss(sample_rng, _model.Observation.from_dict(inputs_2), inputs_2["actions"], train=False)
            loss =jnp.mean(chunked_loss).item()
            outputs["loss"] = loss
            outputs["gt_actions"] = np.asarray(inputs_2["actions"][0, ...])

            # print("GT ACTIONS PROVIDED, Policy is returning the following outputs: ", outputs)
            return outputs
        
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        start_time = time.monotonic()
        self._rng, sample_rng = jax.random.split(self._rng)
        start_time = time.perf_counter()
        # print(f"🤠🤠🤠: Time Started Sample_actions: {start_time}")
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng, _model.Observation.from_dict(inputs), **self._sample_kwargs),
        }
        end_time = time.perf_counter()
        # print(f"🤠🤠🤠: Time Ended Sample_actions: {end_time}")
        # print(f"🤠🤠🤠: Time Taken for Sample_actions: {end_time - start_time}")

        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        model_time = time.monotonic() - start_time

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
