"""Evaluate a policy on a single LIBERO task with out-of-distribution perturbations."""

import argparse
import ast
import collections
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import wandb

from OOD_eval_helper import get_expanded_libero_env

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

MAX_STEPS_PER_SUITE = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LIBERO tasks with OOD perturbations.")
    parser.add_argument("--task_suite_name", type=str, required=True,
                        choices=list(MAX_STEPS_PER_SUITE.keys()),
                        help="Task suite to evaluate")
    parser.add_argument("--task_name", type=str, required=True,
                        help="Name of the task to evaluate (in task suite)")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host for model server")
    parser.add_argument("--port", required=True, type=int, help="Port for model server")
    parser.add_argument("--resize_size", type=int, default=224, help="Resize size for images")
    parser.add_argument("--replan_steps", type=int, default=5,
                        help="Number of steps between replanning")
    parser.add_argument("--num_trials_per_task", type=int, required=True,
                        help="Number of rollouts per task")
    parser.add_argument("--num_steps_wait", type=int, default=10,
                        help="Steps to wait for objects to stabilize")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility")
    parser.add_argument("--results_root_dir", type=str, required=True,
                        help="Root directory for results")

    # OOD perturbation parameters.
    parser.add_argument("--remove_train_distractors", action="store_true",
                        help="Remove non-target objects from the environment")
    parser.add_argument("--permute_objs_of_interest", action="store_true",
                        help="Permute the positions of objects of interest")
    parser.add_argument("--expansion_half_len_factor", type=float, required=True,
                        help="Factor to expand the initial spawn region")
    parser.add_argument("--min_distractors", type=int, required=True,
                        help="Minimum number of distractor objects")
    parser.add_argument("--max_distractors", type=int, required=True,
                        help="Maximum number of distractor objects")
    parser.add_argument("--do_translation", action="store_true",
                        help="Apply fixed translations to objects of interest")
    parser.add_argument("--translation_scales_dict", type=ast.literal_eval, required=True,
                        help="Dict mapping object names to (dx, dy) offsets")
    parser.add_argument("--swap_dict", type=ast.literal_eval, required=True,
                        help="Dict mapping scene element names for background swapping")

    parser.add_argument("--wandb_project", type=str, default="libero_evals",
                        help="W&B project name")
    return parser.parse_args()


def eval_libero_OOD(args):
    """Run OOD evaluation rollouts for a single LIBERO task and log results."""
    assert args.exp_name is not None, "exp_name is required"
    assert args.task_name is not None, "task_name is required"
    # Underscores are used for shell scripting; convert back to spaces for LIBERO.
    task_name = args.task_name.replace("_", " ")

    np.random.seed(args.seed)
    results_dir = (
        f"{args.results_root_dir}/{args.exp_name}/{args.task_suite_name}"
        f"/{task_name}__seed_{args.seed}"
    )
    video_out_path = f"{results_dir}/videos"

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(video_out_path).mkdir(parents=True, exist_ok=True)

    max_steps = MAX_STEPS_PER_SUITE.get(args.task_suite_name)
    if max_steps is None:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    eval_type = "_".join(args.exp_name.split("_")[:2])
    checkpoint_name = "_".join(args.exp_name.split("_")[2:])
    logging.info(f"eval_type: {eval_type}, checkpoint_name: {checkpoint_name}")

    wandb.init(
        resume="allow",
        project=args.wandb_project,
        group=checkpoint_name,
        name=f"{args.task_suite_name}_{task_name}__seed_{args.seed}",
        config={
            "eval_type": eval_type,
            "task_suite": args.task_suite_name,
            "task_name": task_name,
            "exp_name": args.exp_name,
            "resize_size": args.resize_size,
            "replan_steps": args.replan_steps,
            "num_steps_wait": args.num_steps_wait,
            "num_trials_per_task": args.num_trials_per_task,
            "seed": args.seed,
            "remove_train_distractors": args.remove_train_distractors,
            "permute_objs_of_interest": args.permute_objs_of_interest,
            "expansion_half_len_factor": args.expansion_half_len_factor,
            "min_distractors": args.min_distractors,
            "max_distractors": args.max_distractors,
        },
    )

    # Find the target task and create the OOD environment.
    env, task_description = None, None
    for task_id in range(num_tasks_in_suite):
        task = task_suite.get_task(task_id)
        if task.language != task_name:
            continue
        env, task_description = get_expanded_libero_env(
            task,
            args.expansion_half_len_factor,
            ood_only=True,
            min_distractors=args.min_distractors,
            max_distractors=args.max_distractors,
            resolution=LIBERO_ENV_RESOLUTION,
            seed=args.seed,
            distractor_seed=args.seed,
            translation_seed=args.seed,
            do_translation=args.do_translation,
            permute_objs_of_interest=args.permute_objs_of_interest,
            remove_train_distractors=args.remove_train_distractors,
            translation_scales_dict=args.translation_scales_dict,
            swap_dict=args.swap_dict,
        )
        logging.info(f"Created OOD environment for task: {task_name}")
        break

    if env is None:
        wandb.finish()
        raise ValueError(
            f"Task '{task_name}' not found in task suite {args.task_suite_name}"
        )

    total_episodes, total_successes = 0, 0
    for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
        logging.info(f"Task: {task_description}")
        env.reset()
        action_plan = collections.deque()

        t = 0
        replay_images = []
        logging.info(f"Starting episode {episode_idx + 1}...")
        while t < max_steps + args.num_steps_wait:
            try:
                if t < args.num_steps_wait:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(
                    obs["robot0_eye_in_hand_image"][::-1, ::-1]
                )
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                )
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(
                        wrist_img, args.resize_size, args.resize_size
                    )
                )
                replay_images.append(img)
                if not action_plan:
                    element = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate(
                            (
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )
                        ),
                        "prompt": str(task_description),
                    }
                    action_chunk = client.infer(element)["actions"]
                    assert len(action_chunk) >= args.replan_steps, (
                        f"Replan every {args.replan_steps} steps, but policy only "
                        f"predicts {len(action_chunk)} steps."
                    )
                    action_plan.extend(action_chunk[: args.replan_steps])
                action = action_plan.popleft()
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    total_successes += 1
                    break
                t += 1
            except Exception as e:
                logging.error(f"Caught exception: {e}")
                break

        total_episodes += 1
        suffix = "success" if done else "failure"
        task_segment = task_description.replace(" ", "_")
        video_file = (
            pathlib.Path(video_out_path)
            / f"rollout_{task_segment}__episode_{episode_idx}_{suffix}.mp4"
        )
        imageio.mimwrite(
            video_file,
            [np.asarray(x) for x in replay_images],
            fps=25,
        )
        wandb.log(
            {
                "episode_idx": episode_idx,
                "success": int(done),
                "cumulative_success_rate": total_successes / total_episodes,
                "video": wandb.Video(str(video_file), fps=25, format="mp4"),
            }
        )
        logging.info(f"Success: {done}, episode_length: {t}")
        logging.info(f"Episodes completed: {total_episodes}")
        logging.info(
            f"Successes: {total_successes} "
            f"({total_successes / total_episodes * 100:.1f}%)"
        )

    env.close()
    wandb.finish()
    logging.info(
        f"Total success rate: {float(total_successes) / float(total_episodes)}"
    )

    with open(f"{results_dir}/results.txt", "a") as f:
        f.write(f"Exp: {args.exp_name}\n")
        f.write(f"Task suite: {args.task_suite_name}\n")
        f.write(f"Task name: {task_name}\n")
        f.write(f"Total episodes: {total_episodes}\n")
        f.write(f"Total successes: {total_successes}\n")
        f.write(
            f"Total success rate: {float(total_successes) / float(total_episodes)}\n"
        )
        f.write(f"Seed: {args.seed}\n")
    return float(total_successes) / float(total_episodes)


def _quat2axisangle(quat):
    """Convert quaternion to axis-angle representation.

    Adapted from robosuite:
    https://github.com/ARISE-Initiative/robosuite/blob/eafb81f/robosuite/utils/transform_utils.py#L490
    """
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    eval_libero_OOD(args)
