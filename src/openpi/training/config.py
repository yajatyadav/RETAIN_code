import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0 as pi0
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None


    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = True

    # If true, will disable syncing the dataset from the Hugging Face Hub. Allows training on local-only datasets.
    local_files_only: bool = False

    # Only used for RLDS data loader.
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None
    

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    repo_id: str = tyro.MISSING
    assets: AssetsConfig = dataclasses.field(
        default_factory=lambda: AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi0_fast_droid/assets",
            asset_id="droid",
        )
    )
    base_config: DataConfig = dataclasses.field(
        default_factory=lambda: DataConfig(
            local_files_only=True,
            prompt_from_task=True,
            action_sequence_keys=("actions",),
        )
    )
    @override
    def create(self, assets_dir: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
            _transforms.RepackTransform(
                {
                    "observation/exterior_image_1_left": "observation/image_primary",
                    "observation/wrist_image_left": "observation/image_wrist",
                    "state": "observation/proprio",
                    "actions" :"action",
                    "prompt": "task/language_instruction",
                }
            )
            ]
        )
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidRLDSInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dir, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """
    repo_id: str = tyro.MISSING
    
    base_config: DataConfig = dataclasses.field(
        default_factory=lambda: DataConfig(
            local_files_only=True,
            prompt_from_task=True,
            action_sequence_keys=("action",),
        )
    )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation/image_primary",
                        "observation/wrist_image": "observation/image_wrist",
                        "observation/state": "observation/proprio",
                        "actions": "action",
                        "prompt": "task/language_instruction",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # TODO(karl): comment this out once we have updated the Libero checkpoints to not use
        # the delta action transform
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            # use_quantile_norm=model_config.model_type == ModelType.PI0_FAST,
        )



@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # The repack transform is *only* applied to the data coming from the dataset,
        # and *not* during inference. We can use it to make inputs from the dataset look
        # as close as possible to those coming from the inference environment (e.g. match the keys).
        # Below, we match the keys in the dataset (which we defined in the data conversion script) to
        # the keys we use in our inference pipeline (defined in the inference script for libero).
        # For your own dataset, first figure out what keys your environment passes to the policy server
        # and then modify the mappings below so your dataset's keys get matched to those target keys.
        # The repack transform simply remaps key names here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # The data transforms are applied to the data coming from the dataset *and* during inference.
        # Below, we define the transforms for data going into the model (``inputs``) and the transforms
        # for data coming out of the model (``outputs``) (the latter is only used during inference).
        # We defined these transforms in `libero_policy.py`. You can check the detailed comments there for
        # how to modify the transforms to match your dataset. Once you created your own transforms, you can
        # replace the transforms below with your own.
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # One additional data transform: pi0 models are trained on delta actions (relative to the first
        # state in each action chunk). IF your data has ``absolute`` actions (e.g. target joint angles)
        # you can uncomment the following line to convert the actions to delta actions. The only exception
        # is for the gripper actions which are always absolute.
        # In the example below, we would apply the delta conversion to the first 6 actions (joints) and
        # leave the 7th action (gripper) unchanged, i.e. absolute.
        # In Libero, the raw actions in the dataset are already delta actions, so we *do not* need to
        # apply a separate delta conversion (that's why it's commented out). Choose whether to apply this
        # transform based on whether your dataset uses ``absolute`` or ``delta`` actions out of the box.

        # TODO(karl): comment this out once we have updated the Libero checkpoints to not use
        # the delta action transform
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        print("DATA_CONFIG: data_transforms: ", data_transforms)
        print("DATA_CONFIG: model_transforms: ", model_transforms)
        print("DATA_CONFIG: repack_transform: ", repack_transform)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )

@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "su25_generalist_finetuning"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    weight_loader_list: list[weight_loaders.WeightLoader] | None = None
    model_mixing_coefficients: list[float] | None = None
    is_merged_model: bool = False

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    is_RLDS: bool = False
    # For co-training with multiple datasets.
    data_list: list[DataConfigFactory] | None = None
    data_mix_weights: list[float] | None = None
    # Root directory for RLDS-format datasets.
    data_root_dir: str = "/data/openx"

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 16
    # Number of train steps (batches) to run.
    num_train_steps: int = 10_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 2_500
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 1

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # causes us to save some very early checkpoints, [50, 200, 750]
    save_early_checkpoints: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")

LIBERO_GOAL_NUM_TRANSITIONS = 52042
LIBERO_OBJECT_NUM_TRANSITIONS = 66984
LIBERO_SPATIAL_NUM_TRANSITIONS = 52970
LIBERO_LM_NUM_TRANSITIONS = 567494
TOT_NUM_TRANSITIONS = LIBERO_GOAL_NUM_TRANSITIONS + LIBERO_OBJECT_NUM_TRANSITIONS + LIBERO_SPATIAL_NUM_TRANSITIONS + LIBERO_LM_NUM_TRANSITIONS
LIBERO_GOAL_FRACTION_OF_TOTAL = LIBERO_GOAL_NUM_TRANSITIONS / TOT_NUM_TRANSITIONS
LIBERO_OBJECT_FRACTION_OF_TOTAL = LIBERO_OBJECT_NUM_TRANSITIONS / TOT_NUM_TRANSITIONS
LIBERO_SPATIAL_FRACTION_OF_TOTAL = LIBERO_SPATIAL_NUM_TRANSITIONS / TOT_NUM_TRANSITIONS
LIBERO_LM_FRACTION_OF_TOTAL = LIBERO_LM_NUM_TRANSITIONS / TOT_NUM_TRANSITIONS


LIBERO_CONFIGS = [
    
    # for inference only, "libero" is the dataset pi0 was finetuned on; hence when evaling pi0_libero, we compute norm_stats on "libero" and use that during inference
    # all pi0 + libero models should use this for inference!
    TrainConfig(
        name="pi0_libero_MINE",
        model=pi0.Pi0Config(),
        data=LeRobotLiberoDataConfig(repo_id="yajatyadav/libero", base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            )),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
    ),

    TrainConfig(
        name="pi0_all_libero_except_libero_10",
        model=pi0.Pi0Config(),
        data=LeRobotLiberoDataConfig(repo_id="yajatyadav/all_libero_except_libero_10", base_config=DataConfig(
            prompt_from_task=True,
        )),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=200_000,
        save_interval=20_000,
        batch_size=128,
        lr_schedule=_optimizer.ConstantSchedule(
            lr=2e-5,
        ),
    ),

    ## only adding this config so we can use the "OG" eval scripts with a newly trained ckpt i transferred over
    TrainConfig(
        name="pi0_all_libero_but_10_flipped_train_split",
        model=pi0.Pi0Config(),
        data=LeRobotLiberoDataConfig(repo_id="yajatyadav/all_libero_but_10_flipped_train_split", base_config=DataConfig(
            prompt_from_task=True,
        )),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        save_interval=5_000,
        batch_size=64,
    ),

    #
    # Fine-tuning Libero configs.
    #
    # These train configs define the hyperparameters for fine-tuning the base model on your own dataset.
    # They are used to define key elements like the dataset you are training on, the base checkpoint you
    # are using, and other hyperparameters like how many training steps to run or what learning rate to use.
    # For your own dataset, you can copy this class and modify the dataset name, and data transforms based on
    # the comments below.
    TrainConfig(
        name="full_FT_pi0_libero_except_10__turn_on_the_stove_and_put_the_moka_pot_on_it",
        model=pi0.Pi0Config(),
        weight_loader=weight_loaders.CheckpointWeightLoader("/raid/users/yajatyadav/checkpoints/pi0_all_libero_except_libero_10/pi0_all_libero_except_libero_10_MINE__config_will/80000/params"),
        data=LeRobotLiberoDataConfig(
            repo_id="yajatyadav/libero_10_turn_on_the_stove_and_put_the_moka_pot_on_it",
            assets = AssetsConfig(
                    assets_dir = "/raid/users/yajatyadav/checkpoints/pi0_all_libero_except_libero_10/pi0_all_libero_except_libero_10_MINE__config_will/80000/assets",
                    asset_id = "yajatyadav/all_libero_except_libero_10",
                ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),            
        ),
        batch_size=64,
        num_train_steps=5_000,
        save_interval=250,
        log_interval=25,
        save_early_checkpoints=False,
    ),

    TrainConfig(
        name="cotrain_pi0_libero_except_10__turn_on_the_stove_and_put_the_moka_pot_on_it",
        model=pi0.Pi0Config(),
        weight_loader=weight_loaders.CheckpointWeightLoader("/raid/users/yajatyadav/checkpoints/pi0_all_libero_except_libero_10/pi0_all_libero_except_libero_10_MINE__config_will/80000/params"),

        is_RLDS=True,
        data_root_dir = '/raid/users/yajatyadav/tensorflow_datasets',
        data_list = [
            # FT dataset- 10866 transitions
            RLDSLiberoDataConfig(
            repo_id="libero_10_turn_on_the_stove_and_put_the_moka_pot_on_it",
            assets = AssetsConfig(
                assets_dir = "/raid/users/yajatyadav/checkpoints/pi0_all_libero_except_libero_10/pi0_all_libero_except_libero_10_MINE__config_will/80000/assets",
                asset_id = "yajatyadav/all_libero_except_libero_10",
                )
            ),

            # pretraining datasets, data_mix_weights will be prop to each's size
            # goal_no_noops: 52042 transitions
            RLDSLiberoDataConfig(
            repo_id="libero_goal_no_noops",
            assets = AssetsConfig(
                assets_dir = "/raid/users/yajatyadav/checkpoints/pi0_all_libero_except_libero_10/pi0_all_libero_except_libero_10_MINE__config_will/80000/assets",
                asset_id = "yajatyadav/all_libero_except_libero_10",
                )
            ),
            
            # object_no_noops: 66984 transitions
            RLDSLiberoDataConfig(
            repo_id="libero_object_no_noops",
            assets = AssetsConfig(
                assets_dir = "/raid/users/yajatyadav/checkpoints/pi0_all_libero_except_libero_10/pi0_all_libero_except_libero_10_MINE__config_will/80000/assets",
                asset_id = "yajatyadav/all_libero_except_libero_10",
                )
            ),

            # spatial_no_noops: 52970 transitions
            RLDSLiberoDataConfig(
            repo_id="libero_spatial_no_noops",
            assets = AssetsConfig(
                assets_dir = "/raid/users/yajatyadav/checkpoints/pi0_all_libero_except_libero_10/pi0_all_libero_except_libero_10_MINE__config_will/80000/assets",
                asset_id = "yajatyadav/all_libero_except_libero_10",
                )
            ),

            # lm_90: 567,494 transitions
            RLDSLiberoDataConfig(
            repo_id="libero_lm_90",
            assets = AssetsConfig(
                assets_dir = "/raid/users/yajatyadav/checkpoints/pi0_all_libero_except_libero_10/pi0_all_libero_except_libero_10_MINE__config_will/80000/assets",
                asset_id = "yajatyadav/all_libero_except_libero_10",
                )
            ),
        ],
        ## 80/20 mix
        data_mix_weights=[1, 0.25 * LIBERO_GOAL_FRACTION_OF_TOTAL, 0.25 * LIBERO_OBJECT_FRACTION_OF_TOTAL, 0.25 * LIBERO_SPATIAL_FRACTION_OF_TOTAL, 0.25 * LIBERO_LM_FRACTION_OF_TOTAL],

        batch_size=64,
        num_train_steps=5_000,
        save_interval=250,
        log_interval=25,
        save_early_checkpoints=False,
    ),




    TrainConfig(
        name="full_FT_pi0_libero_except_10__put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
        model=pi0.Pi0Config(),
        weight_loader=weight_loaders.CheckpointWeightLoader("/raid/users/yajatyadav/checkpoints/pi0_all_libero_except_libero_10/pi0_all_libero_except_libero_10_MINE__config_will/80000/params"),
        data=LeRobotLiberoDataConfig(
            repo_id="yajatyadav/libero_10_put_the_yellow_and_white_mug_in_the_microwave_and_close_it",
            assets = AssetsConfig(
                    assets_dir = "/raid/users/yajatyadav/checkpoints/pi0_all_libero_except_libero_10/pi0_all_libero_except_libero_10_MINE__config_will/80000/assets",
                    asset_id = "yajatyadav/all_libero_except_libero_10",
                ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),            
        ),
        batch_size=64,
        num_train_steps=5_000,
        save_interval=250,
        log_interval=25,
        save_early_checkpoints=False,
    ),




    TrainConfig(
        # Change the name to reflect your model and dataset.
        name="full_FT_pi0_libero_close_the_microwave",
        # Here you define the model config -- In this example we use pi0 as the model
        # architecture and perform *full* finetuning. in the examples below we show how to modify
        # this to perform *low-memory* (LORA) finetuning and use pi0-FAST as an alternative architecture.
        model=pi0.Pi0Config(),
        weight_loader=weight_loaders.CheckpointWeightLoader("/raid/users/yajatyadav/checkpoints/pi0_libero_MINE/FT_base_pi0_on_libero_debugging/29999/params"),
        # Here you define the dataset you are training on. In this example we use the Libero
        # dataset. For your own dataset, you can change the repo_id to point to your dataset.
        # Also modify the DataConfig to use the new config you made for your dataset above.
        data=LeRobotLiberoDataConfig(
            repo_id="yajatyadav/libero_90_close_the_microwave",
            assets = AssetsConfig(
                assets_dir = "/raid/users/yajatyadav/checkpoints/pi0_libero_MINE/FT_base_pi0_on_libero_debugging/29999/assets",
                asset_id = "yajatyadav/libero",
            ),
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. If set to True, the prompt will show up in
                # a field called ``prompt`` in the input dict. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        batch_size=64,
        # Here you define which pre-trained checkpoint you want to load to initialize the model.
        # This should match the model config you chose above -- i.e. in this case we use the pi0 base model.
        
        # Below you can define other hyperparameters like the learning rate, number of training steps, etc.
        # Check the base TrainConfig class for a full list of available hyperparameters.
        num_train_steps=20_000,
        save_interval=500, # about 3.3 epochs for the microwave task
        log_interval=25,
        save_early_checkpoints=False,
    ),

    TrainConfig(
        name="full_FT_pi0_libero_close_the_microwave_merged_model",
        model=pi0.Pi0Config(),

        is_merged_model=True,
        weight_loader_list=[
            weight_loaders.CheckpointWeightLoader("/raid/users/yajatyadav/checkpoints/full_FT_pi0_libero_close_the_microwave/full_FT_my_pi0_libero_close_the_microwave__batch_64__lr_defaults__num_train_steps_10000/10500/params"),
            weight_loaders.CheckpointWeightLoader("/raid/users/yajatyadav/checkpoints/pi0_libero_MINE/FT_base_pi0_on_libero_debugging/29999/params"),
        ],
        model_mixing_coefficients=[0.25, 0.75],

        data=LeRobotLiberoDataConfig(
            repo_id="yajatyadav/libero_90_close_the_microwave",
            assets = AssetsConfig(
                assets_dir = "/raid/users/yajatyadav/checkpoints/pi0_libero_MINE/FT_base_pi0_on_libero_debugging/29999/assets",
                asset_id = "yajatyadav/libero",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        lr_schedule=_optimizer.ConstantSchedule(
            lr=0,
        ),
        num_workers=16,
        num_train_steps=1,
        log_interval=1,
        save_interval=1,
        save_early_checkpoints=False,
        wandb_enabled=False,
    ),

    TrainConfig(
        name="full_FT_pi0_libero_pick_up_the_cream_cheese_and_put_it_in_the_tray",
        model=pi0.Pi0Config(),
        weight_loader=weight_loaders.CheckpointWeightLoader("/raid/users/yajatyadav/checkpoints/pi0_libero_MINE/FT_base_pi0_on_libero_debugging/29999/params"),
        data=LeRobotLiberoDataConfig(
            repo_id="yajatyadav/libero_90_pick_up_the_cream_cheese_and_put_it_in_the_tray",
            assets = AssetsConfig(
                assets_dir = "/raid/users/yajatyadav/checkpoints/pi0_libero_MINE/FT_base_pi0_on_libero_debugging/29999/assets",
                asset_id = "yajatyadav/libero",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        batch_size=64,
        num_train_steps=900,
        save_interval=125,
        log_interval=50,
        save_early_checkpoints=False,
    ),

    TrainConfig(
        name="full_FT_pi0_libero_pick_up_the_cream_cheese_and_put_it_in_the_tray_merged_model",
        model=pi0.Pi0Config(),
        
        is_merged_model=True,
        weight_loader_list=[
            weight_loaders.CheckpointWeightLoader("/raid/users/yajatyadav/checkpoints/full_FT_pi0_libero_pick_up_the_cream_cheese_and_put_it_in_the_tray/full_FT_my_pi0_libero_pick_up_the_cream_cheese_and_put_it_in_the_tray__batch_64__lr_defaults__num_train_steps_900/250/params"),
            weight_loaders.CheckpointWeightLoader("/raid/users/yajatyadav/checkpoints/pi0_libero_MINE/FT_base_pi0_on_libero_debugging/29999/params"),
        ],
        model_mixing_coefficients=[0.25, 0.75],

        data=LeRobotLiberoDataConfig(
            repo_id="yajatyadav/libero_90_pick_up_the_cream_cheese_and_put_it_in_the_tray",
            assets = AssetsConfig(
                assets_dir = "/raid/users/yajatyadav/checkpoints/pi0_libero_MINE/FT_base_pi0_on_libero_debugging/29999/assets",
                asset_id = "yajatyadav/libero",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        lr_schedule=_optimizer.ConstantSchedule(
            lr=0,
        ),
        num_workers=16,
        num_train_steps=1,
        log_interval=1,
        save_interval=1,
        save_early_checkpoints=False,
        wandb_enabled=False,     
    ),

    TrainConfig(
        name="full_FT_pi0_DSRL_200k_pick_up_the_cream_cheese_and_put_it_in_the_tray",
        model=pi0.Pi0Config(),
        weight_loader=weight_loaders.CheckpointWeightLoader("/raid/users/yajatyadav/checkpoints/pi0_libero_MINE/FT_base_pi0_on_libero_debugging/29999/params"),
        data=LeRobotLiberoDataConfig(
            repo_id="yajatyadav/DSRL_ckpt_200k_pick_up_the_cream_cheese_and_put_it_in_the_tray",
            assets = AssetsConfig(
                assets_dir = "/raid/users/yajatyadav/checkpoints/pi0_libero_MINE/FT_base_pi0_on_libero_debugging/29999/assets",
                asset_id = "yajatyadav/libero",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        batch_size=64,
        num_train_steps=750,
        save_interval=250,
        log_interval=50,
        save_early_checkpoints=False,
    ),


    TrainConfig(
        name="full_FT_pi0_DSRL_200k_pick_up_the_cream_cheese_and_put_it_in_the_tray_merged_model",
        model=pi0.Pi0Config(),

        is_merged_model=True,
        weight_loader_list=[
            weight_loaders.CheckpointWeightLoader("/raid/users/yajatyadav/checkpoints/full_FT_pi0_DSRL_200k_pick_up_the_cream_cheese_and_put_it_in_the_tray/full_FT_my_pi0_DSRL_200k_pick_up_the_cream_cheese_and_put_it_in_the_tray__batch_64__lr_defaults__num_train_steps_750/250/params"),
            weight_loaders.CheckpointWeightLoader("/raid/users/yajatyadav/checkpoints/pi0_libero_MINE/FT_base_pi0_on_libero_debugging/29999/params"),
        ],
        model_mixing_coefficients=[0.25, 0.75],
        data=LeRobotLiberoDataConfig(
            repo_id="yajatyadav/DSRL_ckpt_200k_pick_up_the_cream_cheese_and_put_it_in_the_tray",
            assets = AssetsConfig(
                assets_dir = "/raid/users/yajatyadav/checkpoints/pi0_libero_MINE/FT_base_pi0_on_libero_debugging/29999/assets",
                asset_id = "yajatyadav/libero",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        lr_schedule=_optimizer.ConstantSchedule(
            lr=0,
        ),
        num_workers=16,
        num_train_steps=1,
        log_interval=1,
        save_interval=1,
        save_early_checkpoints=False,
        wandb_enabled=False,
    )
]







FULL_FT_CONFIGS = [
    # Sanity check whether pi0 full-FT is working with lerobot setup.
    TrainConfig(
        name="full_FT_pi0_whiteboard",
        model=pi0.Pi0Config(action_dim=32, action_horizon=10),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_droid/params"),
        data=LeRobotDROIDDataConfig(repo_id="yajatyadav/whiteboard", assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets/checkpoints/pi0_droid/assets",
                asset_id="droid",
            )
        ),
        num_train_steps=200,
        log_interval=10,
        save_interval=999,
        wandb_enabled=False,
        save_early_checkpoints=False,
    ),
    TrainConfig(
        name="full_FT_libero_10_no_noops",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_droid/params"),
        
        is_RLDS=True,
        data_root_dir="/raid/users/yajatyadav/datasets",
        data=RLDSLiberoDataConfig(repo_id="libero_10_no_noops"),
        batch_size= 32 * 4,
        
        num_train_steps = 1_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=int(1_000 * 0.1),
            peak_lr=3e-5,
            decay_steps=1_000,
            decay_lr=2e-6,
        ),        
        save_interval= 250,
    ),

    TrainConfig(
        name="full_FT_plates_RLDS",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_droid/params"),
        
        is_RLDS=True,
        data_root_dir = "/home/yajatyadav/tensorflow_datasets",
        data=RLDSDroidDataConfig(repo_id="plates"),
        batch_size = 32 * 1,

        num_train_steps = 5_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=int(5_000 * 0.1),
            peak_lr=3e-5,
            decay_steps=5_000,
            decay_lr=2e-6,
        ),
        save_interval= 500,
        save_early_checkpoints=True,  
    ),

    TrainConfig(
        name="full_FT_whiteboard_RLDS",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_droid/params"),
        data=RLDSDroidDataConfig(repo_id="whiteboard"),
        is_RLDS=True,
        num_train_steps = 1_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=int(1_000 * 0.1),
            peak_lr=3e-5,
            decay_steps=1_000,
            decay_lr=2e-6,
        ),
        save_interval= 250,
        save_early_checkpoints=False,
    ),
    TrainConfig(
        name="eval_full_FT_whiteboard_on_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        weight_loader=weight_loaders.CheckpointWeightLoader(params_path="/raid/users/yajatyadav/checkpoints/full_FT_whiteboard_RLDS/full_FT_whiteboard_RLDS_default/9999/params"),
        data=RLDSDroidDataConfig(repo_id="droid_alt"),
        lr_schedule=_optimizer.ConstantSchedule(
            lr=0,
        ),
        num_workers=16,
        is_RLDS=True,
        num_train_steps=100_000,
        log_interval=100,
        save_interval=10_000_000_000,
        save_early_checkpoints=False,
    ),
]

LORA_CONFIGS = [
    TrainConfig(
        name="lora_plates_RLDS",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10, paligemma_variant="gemma_2b_lora"),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_droid/params"),
        freeze_filter=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10, paligemma_variant="gemma_2b_lora").get_freeze_filter(),
        ema_decay=None,
        
        is_RLDS=True,
        data=RLDSDroidDataConfig(repo_id="plates"),
        batch_size=32 * 4,

        num_train_steps=10_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=int(10_000 * 0.1),
            peak_lr=3e-5,
            decay_steps=10_000,
            decay_lr=2e-6,
        ),
        save_interval= 1_000,
        save_early_checkpoints=True,
    ),


    TrainConfig(
        name="lora_whiteboard_RLDS",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10, paligemma_variant="gemma_2b_lora", freeze_vision=True),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_droid/params"),
        freeze_filter=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10, paligemma_variant="gemma_2b_lora", freeze_vision=True).get_freeze_filter(),
        ema_decay=None,
        data=RLDSDroidDataConfig(repo_id="whiteboard"),
        is_RLDS=True,
        num_train_steps = 5_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=int(5_000 * 0.1),
            peak_lr=6e-6,
            decay_steps=5_000,
            decay_lr=4e-7,
        ),
        save_interval= 1_000,
        save_early_checkpoints=True,
    ),
    TrainConfig(
        name="eval_lora_whiteboard_on_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10, paligemma_variant="gemma_2b_lora"),
        weight_loader=weight_loaders.CheckpointWeightLoader(params_path="/raid/users/yajatyadav/checkpoints/lora_whiteboard_RLDS/lora_whiteboard_RLDS_default/9999/params"),
        freeze_filter=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10, paligemma_variant="gemma_2b_lora").get_freeze_filter(),
        ema_decay=None,
        data=RLDSDroidDataConfig(repo_id="droid_alt"),
        lr_schedule=_optimizer.ConstantSchedule(
            lr=0,
        ),
        num_workers=16,
        is_RLDS=True,
        num_train_steps=100_000,
        log_interval=100,
        save_interval=10_000_000_000,
        save_early_checkpoints=False,
    ),
]

VISION_FROZEN_CONFIGS = [
    TrainConfig(
        name="vision_frozen_full_FT_whiteboard_RLDS",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10, freeze_vision=True),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_droid/params"),
        freeze_filter=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10, freeze_vision=True).get_freeze_filter(),
        data=RLDSDroidDataConfig(repo_id="whiteboard"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=30_000,
            decay_lr=2.5e-6,
        ),
        is_RLDS=True,
    ),
    TrainConfig(
        name="eval_vision_frozen_full_FT_whiteboard_on_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10, freeze_vision=True),
        weight_loader=weight_loaders.CheckpointWeightLoader(params_path="/raid/users/yajatyadav/checkpoints/vision_frozen_full_FT_whiteboard_RLDS/vision_frozen_full_FT_whiteboard_RLDS_default/9999/params"),
        freeze_filter=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10, freeze_vision=True).get_freeze_filter(),
        data=RLDSDroidDataConfig(repo_id="droid_alt"),
        lr_schedule=_optimizer.ConstantSchedule(
            lr=0,
        ),
        num_workers=16,
        is_RLDS=True,
        num_train_steps=100_000,
        log_interval=100,
        save_interval=10_000_000_000,
        save_early_checkpoints=False,
    ),
]

COTRAINING_CONFIGS = [
    TrainConfig(
        name="full_FT_pi0_whiteboard_ALT",
        model=pi0.Pi0Config(action_horizon=15, pi05=False),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets-preview/checkpoints/pi0_droid/params"),
        is_RLDS=True,        
        data=RLDSDroidDataConfig(
                repo_id="whiteboard", 
                assets=AssetsConfig(
                    assets_dir="gs://openpi-assets-preview/checkpoints/pi0_droid/assets",
                    asset_id="droid",
                    )
                ),
      
        wandb_enabled=False
    ),
    TrainConfig(
        name="full_FT_pi05_whiteboard",
        model=pi0.Pi0Config(pi05=True, action_horizon=16, action_dim=32),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets-preview/checkpoints/pi05_droid/params"),
        is_RLDS=True,        
        data=RLDSDroidDataConfig(
                repo_id="whiteboard", 
                assets=AssetsConfig(
                    assets_dir="gs://openpi-assets-preview/checkpoints/pi05_droid/assets",
                    asset_id="droid",
                    )
                ),
        num_train_steps = 10_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=3e-6,
            decay_steps=10_000,
            decay_lr=2e-8,
        ),
        batch_size=32,
        save_interval = 500,
        log_interval=25,
        save_early_checkpoints = False,
    ),

    TrainConfig(
        name="cotrain_pi05_whiteboard",
        model=pi0.Pi0Config(pi05=True, action_horizon=16, action_dim=32),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets-preview/checkpoints/pi05_droid/params"),

        is_RLDS=True,
        data_list = [
            RLDSDroidDataConfig(
                repo_id="whiteboard", 
                assets=AssetsConfig(
                    assets_dir="gs://openpi-assets-preview/checkpoints/pi05_droid/assets",
                    asset_id="droid",
                    )
                ),
            RLDSDroidDataConfig(
                repo_id="droid_alt", 
                assets=AssetsConfig(
                    assets_dir="gs://openpi-assets-preview/checkpoints/pi05_droid/assets",
                    asset_id="droid",
                    )
                ),
            ],
        data_mix_weights=[1.0, 0.25],
        batch_size=32,
        num_workers=16,

        num_train_steps = 20_000,
        save_interval = 1_000,
        save_early_checkpoints = True,        
    ),

    TrainConfig(
        # only for debuggin purposes, hence no checkpointing and logging more frequently
        name="full_FT_droid_RLDS",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_droid/params"),
        data=RLDSDroidDataConfig(repo_id="droid_alt"),
        lr_schedule=_optimizer.ConstantSchedule(
            lr=2.5e-5,
        ),
        num_workers=16,
        is_RLDS=True,
        # num_train_steps=10_000,
        # log_interval=100,
        # save_interval=2500,
        # save_early_checkpoints=True,
    ),

    TrainConfig(
        name="plates_cotrain",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_droid/params"),
        
        is_RLDS=True,
        data_list = [
            RLDSDroidDataConfig(repo_id="plates"),
            RLDSDroidDataConfig(repo_id="droid_alt")
        ],
        data_mix_weights= [1, 4],
        batch_size = 32 * 4,

        num_train_steps = 10_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=int(10_000 * 0.1),
            peak_lr=3e-5,
            decay_steps=10_000,
            decay_lr=2e-6,
        ),
        save_interval = 1_000,
        save_early_checkpoints = True,
    ),


    TrainConfig(
        name="whiteboard_cotrain",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_droid/params"),
        is_RLDS=True,
        data_list = [
            RLDSDroidDataConfig(repo_id="whiteboard"), 
            RLDSDroidDataConfig(repo_id="droid_alt"),
            ],
        data_mix_weights=[1.0, 4.0], ## 20-80 mixture
        batch_size=32 * 4,
        num_workers=24,
        num_train_steps = 15_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=int(15_000 * 0.1),
            peak_lr=3e-5,
            decay_steps=15_000,
            decay_lr=2e-6,
        ),
        save_interval= 1_000
    ),
    TrainConfig(
        name="eval_whiteboard_cotrain_on_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        weight_loader=weight_loaders.CheckpointWeightLoader(params_path="/raid/users/yajatyadav/checkpoints/whiteboard_cotrain/whiteboard_cotrain_80_20_default/9999/params"),
        data=RLDSDroidDataConfig(repo_id="droid_alt"),
        lr_schedule=_optimizer.ConstantSchedule(
            lr=0,
        ),
        num_workers=16,
        is_RLDS=True,
        num_train_steps=100_000,
        log_interval=100,
        save_interval=10_000_000_000,
        save_early_checkpoints=False,
    ),



    # TrainConfig(
    #     name="plates_cotrain",
    #     model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
    #     weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_droid/params"),
    #     data_list = [
    #         RLDSDroidDataConfig(repo_id="plates"), 
    #         RLDSDroidDataConfig(repo_id="droid_alt"),
    #         ],
    #     data_mix_weights=[1.0, 0.25], ## 80-20 mixture
    #     num_workers=24,
    #     is_RLDS=True,
    # ),
]

MISC_CONFIGS = [

    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="yajatyadav/whiteboard",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="gs://openpi-assets-preview/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets-preview/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        save_interval=500,
        log_interval=25,
        batch_size=64,
        save_early_checkpoints=False,
    ),


    ## the next couple configs are inference-only

    TrainConfig(
        name="pi05_droid",
        model=pi0.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),

    TrainConfig(
        name="pi0_droid",
        model=pi0.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),

    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    ]

MODEL_MERGING_CONFIGS = [

    ## interploating b/w pi0_fast and full_FT
    TrainConfig(
        name="full_FT_whiteboard_merged_model",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),

        is_merged_model=True,
        weight_loader_list=[weight_loaders.CheckpointWeightLoader(params_path="/raid/users/yajatyadav/checkpoints/full_FT_whiteboard_RLDS/full_FT_whiteboard_RLDS__lr_suraj__num_train_steps__5000/500/params"),
                            weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_droid/params")
                            ],
        model_mixing_coefficients=[0.75, 0.25],
        
        data=RLDSDroidDataConfig(repo_id="whiteboard"),
        is_RLDS=True,
        lr_schedule=_optimizer.ConstantSchedule(
            lr=0,
        ),
        num_workers=16,
        num_train_steps=1,
        log_interval=1,
        save_interval=1,
        save_early_checkpoints=False,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="full_FT_plates_merged_model",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),

        is_merged_model=True,
        weight_loader_list=[weight_loaders.CheckpointWeightLoader(params_path="/raid/users/yajatyadav/checkpoints/full_FT_plates_RLDS/full_FT_plates_RLDS__batch_size_32__lr_suraj__num_train_steps__5000/1500/params"),
                            weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_droid/params")
                            ],
        model_mixing_coefficients=[0.75, 0.25],

        data=RLDSDroidDataConfig(repo_id="plates"),
        is_RLDS=True,
        lr_schedule=_optimizer.ConstantSchedule(
            lr=0,
        ),
        num_workers=16,
        num_train_steps=1,
        log_interval=1,
        save_interval=1,
        save_early_checkpoints=False,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="DEBUG_eval_full_FT_whiteboard_merged_model_on_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        
        is_merged_model=True,
        weight_loader_list=[weight_loaders.CheckpointWeightLoader(params_path="/raid/users/yajatyadav/checkpoints/full_FT_whiteboard_RLDS/full_FT_whiteboard_RLDS_default/9999/params"),
                            weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_droid/params")
                            ],
        model_mixing_coefficients=[0.75, 0.25],
        
        data=RLDSDroidDataConfig(repo_id="droid_alt"),
        is_RLDS=True,
        
        lr_schedule=_optimizer.ConstantSchedule(
            lr=0,
        ),
        num_workers=16,
        num_train_steps=100_000,
        log_interval=10,
        save_interval=10_000_000_000,
        save_early_checkpoints=False,
    ),


    ### now interpolating b/w pi0_fast and cotrained checkpoint
    TrainConfig(
        name="cotrain_whiteboard_merged_model",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),

        is_merged_model=True,
        weight_loader_list=[weight_loaders.CheckpointWeightLoader(params_path="/raid/users/yajatyadav/checkpoints/whiteboard_cotrain/whiteboard_cotrain_20_80__lr_suraj___num_train_steps_15000/12500/params"),
                            weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_droid/params")
                            ],
        model_mixing_coefficients=[0.75, 0.25],
        
        data=RLDSDroidDataConfig(repo_id="whiteboard"),
        is_RLDS=True,
        lr_schedule=_optimizer.ConstantSchedule(
            lr=0,
        ),
        num_workers=16,
        num_train_steps=1,
        log_interval=1,
        save_interval=1,
        save_early_checkpoints=False,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="cotrain_plates_merged_model",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),

        is_merged_model=True,
        weight_loader_list=[weight_loaders.CheckpointWeightLoader(params_path="/raid/users/yajatyadav/checkpoints/plates_cotrain/plates_cotrain_80_20__batch_size_32__lr_suraj__num_train_steps_50000/5000/params"),
                            weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_droid/params")
                            ],
        model_mixing_coefficients=[0.25, 0.75],
        
        data=RLDSDroidDataConfig(repo_id="plates"),
        is_RLDS=True,
        lr_schedule=_optimizer.ConstantSchedule(
            lr=0,
        ),
        num_workers=16,
        num_train_steps=1,
        log_interval=1,
        save_interval=1,
        save_early_checkpoints=False,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="DEBUG_eval_cotrain_whiteboard_merged_model_on_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        
        is_merged_model=True,
        weight_loader_list=[weight_loaders.CheckpointWeightLoader(params_path="/raid/users/yajatyadav/checkpoints/whiteboard_cotrain/whiteboard_cotrain_80_20_default/9999/params"),
                            weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_droid/params")
                            ],
        model_mixing_coefficients=[1.0, 0.0],
        
        data=RLDSDroidDataConfig(repo_id="droid_alt"),
        is_RLDS=True,
        
        lr_schedule=_optimizer.ConstantSchedule(
            lr=0,
        ),
        num_workers=16,
        num_train_steps=100_000,
        log_interval=10,
        save_interval=10_000_000_000,
        save_early_checkpoints=False,
    ),



    ## now interpolating b/w full_FT and lora by scaling the LoRA-specific parameters
    TrainConfig(
        name="DEBUG_eval_lora_whiteboard_merged_model_on_whiteboard",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10, paligemma_variant="gemma_2b_lora"),
        ## these don't really matter, as we are not training the model
        freeze_filter=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10, paligemma_variant="gemma_2b_lora").get_freeze_filter(),
        ema_decay=None,


        is_merged_model=True,
        weight_loader_list=[weight_loaders.CheckpointWeightLoader(params_path="/raid/users/yajatyadav/checkpoints/lora_whiteboard_RLDS/lora_whiteboard_RLDS_default/9999/params"),
                            weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_droid/params")
                            ],
        model_mixing_coefficients=[0.75, 0.25], # 2 nums needed to be consistent, even though only the first one is used
        
        data=RLDSDroidDataConfig(repo_id="whiteboard"),
        is_RLDS=True,

        lr_schedule=_optimizer.ConstantSchedule(
            lr=0,
        ),
        num_workers=16,
        num_train_steps=100_000,
        log_interval=10,
        save_interval=10_000_000_000,
        save_early_checkpoints=False,
    ),
    TrainConfig(
        name="DEBUG_eval_lora_whiteboard_merged_model_on_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10, paligemma_variant="gemma_2b_lora"),
        ## these don't really matter, as we are not training the model
        freeze_filter=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10, paligemma_variant="gemma_2b_lora").get_freeze_filter(),
        ema_decay=None,
        
        is_merged_model=True,
        weight_loader_list=[weight_loaders.CheckpointWeightLoader(params_path="/raid/users/yajatyadav/checkpoints/lora_whiteboard_RLDS/lora_whiteboard_RLDS_default/9999/params"),
                            weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_droid/params")
                            ],
        model_mixing_coefficients=[0.75, 0.25],
        
        data=RLDSDroidDataConfig(repo_id="droid_alt"),
        is_RLDS=True,
        
        lr_schedule=_optimizer.ConstantSchedule(
            lr=0,
        ),
        num_workers=16,
        num_train_steps=100_000,
        log_interval=10,
        save_interval=10_000_000_000,
        save_early_checkpoints=False,
    )
]

FROM_SCRATCH_CONFIGS = [
    TrainConfig(
        name="paligemma_full_FT_whiteboard_RLDS",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        weight_loader = weight_loaders.PaliGemmaWeightLoader(),
        
        is_RLDS=True,
        data=RLDSDroidDataConfig(repo_id="whiteboard"),
        batch_size=32 * 4,
        
        num_train_steps=30_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=int(30_000 * 0.1),
            peak_lr=3e-4,
            decay_steps=30_000,
            decay_lr=2e-5,
        ),
        save_interval=3_000,
    ),

    TrainConfig(
        name="paligemma_full_FT_plates_RLDS",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        weight_loader = weight_loaders.PaliGemmaWeightLoader(),
        
        is_RLDS=True,
        data=RLDSDroidDataConfig(repo_id="plates"),
        batch_size=32 * 4,

        num_train_steps=30_000,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=int(30_000 * 0.1),
            peak_lr=3e-4,
            decay_steps=30_000,
            decay_lr=2e-5,
        ),
        save_interval=3_000,
    )
]

_CONFIGS = FULL_FT_CONFIGS + LORA_CONFIGS + COTRAINING_CONFIGS + VISION_FROZEN_CONFIGS + MODEL_MERGING_CONFIGS + MISC_CONFIGS + FROM_SCRATCH_CONFIGS + LIBERO_CONFIGS


if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]