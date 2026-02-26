"""Utilities for creating out-of-distribution LIBERO evaluation environments.

Supports three kinds of perturbations that can be composed:
  1. **Translation** -- shift the initial spawn regions of objects of interest.
  2. **Expansion** -- enlarge the spawn regions so objects may appear in novel positions.
  3. **Distractors** -- add randomly chosen unseen objects to the scene.
"""

import logging
import os

import gym
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.benchmark.mu_creation import *  # noqa: F401,F403
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.bddl_base_domain import BDDLUtils
from libero.libero.envs.objects import get_object_dict
from libero.libero.utils import task_generation_utils
from libero.libero.utils.bddl_generation_utils import (
    get_object_dict as get_object_num_dict,
    get_xy_region_kwargs_list_from_regions_info,
)
from libero.libero.utils.mu_utils import get_scene_class, get_scene_dict, register_mu
from libero.libero.utils.task_generation_utils import (
    generate_bddl_from_task_info,
    register_task_info,
)

logger = logging.getLogger(__name__)

# Objects too large to serve as distractors (would occlude the scene).
_LARGE_OBJECTS = frozenset({
    "basket",
    "basin_faucet",
    "chefmate_8_frypan",
    "desk_caddy",
    "dining_set_group",
    "faucet",
    "flat_stove",
    "microwave",
    "rack",
    "short_cabinet",
    "short_fridge",
    "slide_cabinet",
    "white_cabinet",
    "white_storage_box",
    "window",
    "wine_rack",
    "wooden_cabinet",
    "wooden_shelf",
    "wooden_tray",
    "wooden_two_layer_shelf",
})

_EXCLUDE_OBJECTS = _LARGE_OBJECTS | {
    "cherries", "corn", "mayo", "salad_dressing", "target_zone",
}

_MAX_DISTRACTOR_PLACEMENT_ATTEMPTS = 10_000_000


def generate_mu_with_distractor_objects(
    mu_cls, min_distractors, max_distractors, distractor_seed
):
    """Create a subclass of *mu_cls* that adds random distractor objects.

    The number and identity of distractors are fixed for the lifetime of the
    returned class (determined by *distractor_seed*).  To get different
    distractors, call this function again with a different seed.
    """
    assert max_distractors >= min_distractors

    rng = np.random.default_rng(distractor_seed)
    all_object_categories = set(get_object_dict().keys())
    mu = mu_cls()
    categories_in_use = set(mu.fixture_object_dict.keys()) | set(
        mu.movable_object_dict.keys()
    )
    remaining_objects = sorted(
        list(all_object_categories - categories_in_use - _EXCLUDE_OBJECTS)
    )

    num_distractors = rng.integers(min_distractors, max_distractors + 1)
    selected_objects = list(
        rng.choice(remaining_objects, num_distractors, replace=False)
    )
    logger.info(
        f"Selected {num_distractors} distractor(s) from "
        f"{len(remaining_objects)} candidates: {selected_objects}"
    )

    class MuWithDistractorObjects(mu_cls):
        def __init__(self, *args, **kwargs):

            # Temporarily set this to False so superclass can initialize properly
            self._include_distractor_in_init_states = False
            super().__init__(*args, **kwargs)
            self._include_distractor_in_init_states = True

            self.distractor_objects = selected_objects
            distractor_object_num_info = {
                category: 1 for category in self.distractor_objects
            }
            self.movable_object_dict.update(
                get_object_num_dict(distractor_object_num_info)
            )
            self.define_regions()

        @property
        def init_states(self):
            if self._include_distractor_in_init_states:
                states = super().init_states
                for cat in self.distractor_objects:
                    assert len(self.movable_object_dict[cat]) == 1
                    obj_name = self.movable_object_dict[cat][0]
                    states.append(
                        ("On", obj_name, f"{self.workspace_name}_{cat}_init_region")
                    )
                return states
            else:
                return super().init_states

        def define_regions(self):
            def _is_point_in_ranges(x, y, ranges):
                return any(
                    x1 <= x <= x2 and y1 <= y <= y2
                    for x1, y1, x2, y2 in ranges
                )

            def _sample_point_outside_ranges(ranges, xlim, ylim):
                for attempt in range(_MAX_DISTRACTOR_PLACEMENT_ATTEMPTS):
                    x = rng.uniform(*xlim)
                    y = rng.uniform(*ylim)
                    if not _is_point_in_ranges(x, y, ranges):
                        return x, y
                    if attempt > 0 and attempt % 100_000 == 0:
                        logger.warning(
                            "Still searching for valid distractor placement..."
                        )
                raise ValueError(
                    "Could not find a valid placement outside existing regions"
                )

            if self._include_distractor_in_init_states:
                super().define_regions()
                for cat in self.distractor_objects:
                    assert len(self.movable_object_dict[cat]) == 1
                    occupied_ranges = []
                    for region in self.xy_region_kwargs_list:
                        if "init_region" not in region["region_name"]:
                            continue
                        radius = 0.15
                        if any(
                            lg in region["region_name"] for lg in _LARGE_OBJECTS
                        ):
                            radius = 0.25
                        x1, y1, x2, y2 = region["ranges"][0]
                        occupied_ranges.append(
                            (x1 - radius, y1 - radius, x2 + radius, y2 + radius)
                        )
                    x, y = _sample_point_outside_ranges(
                        occupied_ranges, xlim=(-0.3, 0.3), ylim=(-0.275, 0.275)
                    )
                    self.regions.update(
                        self.get_region_dict(
                            region_centroid_xy=[x, y],
                            region_name=f"{cat}_init_region",
                            target_name=self.workspace_name,
                            region_half_len=0.00001,
                        )
                    )
                    self.xy_region_kwargs_list = (
                        get_xy_region_kwargs_list_from_regions_info(self.regions)
                    )
            else:
                super().define_regions()

    return MuWithDistractorObjects


def edit_bddl_file_with_swap_objects(bddl_file, swap_objects_dict):
    """Perform string replacements in a BDDL file (e.g. to swap backgrounds)."""
    with open(bddl_file, "r") as f:
        content = f.read()
    for original, replacement in swap_objects_dict.items():
        content = content.replace(original, replacement)
    with open(bddl_file, "w") as f:
        f.write(content)


def generate_translated_mu(mu_cls, obj_of_interest, translation_scales_dict, translation_seed):
    """Create a subclass of *mu_cls* that translates object spawn regions."""

    class TranslatedMu(mu_cls):
        def __init__(self, *args, **kwargs):
            self.obj_of_interest_to_region_map = {}
            super().__init__(*args, **kwargs)

        def define_regions(self):
            super().define_regions()
            for obj, (dx, dy) in translation_scales_dict.items():
                for condition in self.init_states:
                    if (
                        condition[1] == obj
                        and condition[0] == "On"
                        and condition[2].endswith("init_region")
                    ):
                        region_name = condition[2].replace(
                            self.workspace_name + "_", ""
                        )
                        current_range = self.regions[region_name]["ranges"]
                        assert len(current_range) == 1
                        x1, y1, x2, y2 = current_range[0]
                        new_range = [(x1 + dx, y1 + dy, x2 + dx, y2 + dy)]
                        logger.debug(
                            f"Translating {obj}: dx={dx}, dy={dy}, "
                            f"range {current_range[0]} -> {new_range[0]}"
                        )
                        self.regions[region_name]["ranges"] = new_range
                        self.obj_of_interest_to_region_map[obj] = new_range
            self.xy_region_kwargs_list = (
                get_xy_region_kwargs_list_from_regions_info(self.regions)
            )

    return TranslatedMu


def generate_expanded_mu(
    mu_cls, expansion_obj_of_interest, expansion_half_len_factor,
    remove_train_distractors=False,
):
    """Create a subclass of *mu_cls* with expanded spawn regions."""

    class ExpandedMu(mu_cls):
        def __init__(self, *args, **kwargs):
            self.obj_of_interest_to_region_map = {}
            super().__init__(*args, **kwargs)

        def define_regions(self):
            super().define_regions()
            for obj in expansion_obj_of_interest:
                for condition in self.init_states:
                    if (
                        condition[1] == obj
                        and condition[0] == "On"
                        and condition[2].endswith("init_region")
                    ):
                        region_name = condition[2].replace(
                            self.workspace_name + "_", ""
                        )
                        current_range = self.regions[region_name]["ranges"]
                        assert len(current_range) == 1
                        x1, y1, x2, y2 = current_range[0]
                        w = x2 - x1
                        h = y2 - y1
                        f = expansion_half_len_factor
                        new_range = [(x1 - f * w, y1 - f * h, x2 + f * w, y2 + f * h)]
                        self.regions[region_name]["ranges"] = new_range
                        self.obj_of_interest_to_region_map[obj] = new_range

            if remove_train_distractors:
                for region in list(self.regions.keys()):
                    obj_name = region.split("_init_region")[0]
                    is_of_interest = any(
                        obj.startswith(obj_name) for obj in expansion_obj_of_interest
                    )
                    if not is_of_interest:
                        del self.regions[region]
            self.xy_region_kwargs_list = (
                get_xy_region_kwargs_list_from_regions_info(self.regions)
            )

    return ExpandedMu


def generate_expanded_mu_permute(
    mu_cls, expansion_obj_of_interest, expansion_half_len_factor,
    remove_train_distractors=False,
):
    """Like generate_expanded_mu, but swaps the spawn regions of two objects.

    Only supports exactly 2 objects of interest.
    """

    class ExpandedMu(mu_cls):
        def __init__(self, *args, **kwargs):
            self.obj_of_interest_to_region_map = {}
            super().__init__(*args, **kwargs)

        def define_regions(self):
            super().define_regions()

            # Iterate through objects of interest
            region_names = {}
            new_ranges_list = []
            for obj in expansion_obj_of_interest:
                for condition in self.init_states:
                    if (
                        condition[1] == obj
                        and condition[0] == "On"
                        and condition[2].endswith("init_region")
                    ):
                        region_name = condition[2].replace(
                            self.workspace_name + "_", ""
                        )
                        region_names[obj] = region_name
                        current_range = self.regions[region_name]["ranges"]
                        assert len(current_range) == 1
                        x1, y1, x2, y2 = current_range[0]
                        w = x2 - x1
                        h = y2 - y1
                        f = expansion_half_len_factor
                        new_ranges_list.append(
                            [(x1 - f * w, y1 - f * h, x2 + f * w, y2 + f * h)]
                        )

            assert len(expansion_obj_of_interest) == 2, (
                "generate_expanded_mu_permute only supports exactly 2 objects"
            )
            # Swap the expanded regions between the two objects.
            swap = {0: 1, 1: 0}
            for i, obj in enumerate(expansion_obj_of_interest):
                for condition in self.init_states:
                    if (
                        condition[1] == obj
                        and condition[0] == "On"
                        and condition[2].endswith("init_region")
                    ):
                        region_name = condition[2].replace(
                            self.workspace_name + "_", ""
                        )
                        self.regions[region_name]["ranges"] = new_ranges_list[swap[i]]
                        self.obj_of_interest_to_region_map[obj] = new_ranges_list[
                            swap[i]
                        ]

            if remove_train_distractors:
                for region in list(self.regions.keys()):
                    obj_name = region.split("_init_region")[0]
                    is_of_interest = any(
                        obj.startswith(obj_name) for obj in expansion_obj_of_interest
                    )
                    if not is_of_interest:
                        del self.regions[region]
            self.xy_region_kwargs_list = (
                get_xy_region_kwargs_list_from_regions_info(self.regions)
            )

    return ExpandedMu


def generate_ood_init_wrapper(
    expanded_mu_cls, expansion_obj_of_interest, expansion_half_len_factor
):
    """Create a gym wrapper that resamples until an object lands outside its original region.

    This ensures at least one object of interest is initialized in the
    *expanded* part of its spawn region (i.e. genuinely out-of-distribution).
    """

    class OODInitWrapper(gym.Wrapper):
        def __init__(self, env):
            super().__init__(env)
            self.env = env
            mu = expanded_mu_cls()
            self.obj_of_interest_to_region_map = mu.obj_of_interest_to_region_map

        def reset(self, **kwargs):
            while True:
                logger.debug("Resampling objects and fixtures...")
                out = self.env.reset(**kwargs)
                for obj_name in expansion_obj_of_interest:
                    obj_state = self.env.env.object_states_dict[obj_name]
                    obj_x, obj_y = obj_state.get_geom_state()["pos"][:2]
                    if obj_name not in self.obj_of_interest_to_region_map:
                        logger.debug(f"Skipping {obj_name} (no init region)")
                        continue
                    init_range = self.obj_of_interest_to_region_map[obj_name]
                    x1_, y1_, x2_, y2_ = init_range[0]
                    w = (x2_ - x1_) / (1 + 2 * expansion_half_len_factor)
                    h = (y2_ - y1_) / (1 + 2 * expansion_half_len_factor)
                    cx, cy = (x1_ + x2_) / 2, (y1_ + y2_) / 2
                    x1, x2 = cx - w / 2, cx + w / 2
                    y1, y2 = cy - h / 2, cy + h / 2
                    if obj_x < x1 or obj_x > x2 or obj_y < y1 or obj_y > y2:
                        return out

    return OODInitWrapper


def get_expanded_libero_env(
    task,
    expansion_half_len_factor,
    ood_only,
    min_distractors,
    max_distractors,
    seed,
    distractor_seed,
    translation_seed,
    translation_scales_dict=None,
    do_translation=False,
    permute_objs_of_interest=False,
    remove_train_distractors=False,
    resolution=256,
    swap_dict=None,
):
    """Build a LIBERO environment with optional OOD perturbations.

    Perturbations are applied in order: translation -> expansion -> distractors.

    Args:
        task: A LIBERO task object.
        expansion_half_len_factor: Factor to expand spawn regions (0 = no expansion).
        ood_only: If True, resample until at least one object is in the expanded
            (non-original) part of its region.
        min_distractors: Minimum distractor objects to add (0 = none).
        max_distractors: Maximum distractor objects to add.
        seed: Seed for the environment and object positions.
        distractor_seed: Seed for distractor selection and placement.
        translation_seed: Seed for translation (currently unused, reserved).
        translation_scales_dict: Mapping of object name to (dx, dy) offsets.
        do_translation: Whether to apply translations.
        permute_objs_of_interest: Whether to swap spawn regions of objects.
        remove_train_distractors: Whether to remove non-target objects.
        resolution: Render resolution.
        swap_dict: Mapping for background/scene element swapping in the BDDL file.

    Returns:
        Tuple of (env, task_description).
    """
    if translation_scales_dict is None:
        translation_scales_dict = {}
    if swap_dict is None:
        swap_dict = {}

    bddl_files_default_path = get_libero_path("bddl_files")
    bddl_file = os.path.join(
        bddl_files_default_path, task.problem_folder, task.bddl_file
    )
    parsed = BDDLUtils.robosuite_parse_problem(bddl_file)

    language = benchmark.grab_language_from_filename(task.bddl_file)
    scene_name = (
        task.bddl_file.replace("_" + language.replace(" ", "_"), "").replace(
            ".bddl", ""
        )
    )

    mu_cls = get_scene_class(scene_name)
    mu_cls_name = mu_cls.__name__
    obj_of_interest = parsed["obj_of_interest"]
    goal_states = [tuple(g) for g in parsed["goal_state"]]

    # Apply perturbations in order: translation -> expansion -> distractors.
    new_mu_cls = mu_cls

    if do_translation:
        new_mu_cls = generate_translated_mu(
            new_mu_cls, obj_of_interest, translation_scales_dict, translation_seed
        )

    if expansion_half_len_factor > 0:
        if permute_objs_of_interest:
            new_mu_cls = generate_expanded_mu_permute(
                new_mu_cls, obj_of_interest, expansion_half_len_factor,
                remove_train_distractors=remove_train_distractors,
            )
        else:
            new_mu_cls = generate_expanded_mu(
                new_mu_cls, obj_of_interest, expansion_half_len_factor,
                remove_train_distractors=remove_train_distractors,
            )

    if min_distractors > 0:
        new_mu_cls = generate_mu_with_distractor_objects(
            new_mu_cls, min_distractors, max_distractors, distractor_seed
        )

    new_mu_cls.__name__ = mu_cls_name + "Expanded"

    scene_dict = get_scene_dict()
    scene_type = [key for key, value in scene_dict.items() if mu_cls in value][0]

    task_generation_utils.TASK_INFO = {}
    register_mu(scene_type=scene_type)(new_mu_cls)

    register_task_info(
        language=language,
        scene_name=scene_name + "_EXPANDED",
        objects_of_interest=obj_of_interest,
        goal_states=goal_states,
    )

    # Use per-seed temp folder to avoid race conditions in parallel runs.
    generate_bddl_from_task_info(folder=f"/tmp/pddl_{seed}")
    task_bddl_file = (
        f"/tmp/pddl_{seed}/{scene_name}_EXPANDED_"
        f"{task.language.replace(' ', '_')}.bddl"
    )

    if swap_dict:
        edit_bddl_file_with_swap_objects(task_bddl_file, swap_dict)

    task_description = task.language
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)

    if ood_only and expansion_half_len_factor > 0:
        ood_init_wrapper_cls = generate_ood_init_wrapper(
            new_mu_cls, obj_of_interest, expansion_half_len_factor
        )
        env = ood_init_wrapper_cls(env)

    logger.info("Initialized OOD LIBERO environment")
    return env, task_description
