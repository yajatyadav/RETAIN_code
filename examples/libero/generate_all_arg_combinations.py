"""Generate evaluation job command lines for ID, generalist, and OOD LIBERO evals.

Produces a list of ``python <script> <args>`` strings that can be distributed
across GPUs for parallel evaluation.
"""

import argparse
import logging
import os

ID_GENERALIST_RUN_FILE = "test_on_libero_task.py"
OOD_RUN_FILE = "test_on_OOD_libero_task.py"

RESULTS_ROOT_DIR = "./"

# In-distribution evaluation config.
EVAL_TASK_SUITE = "libero_10"
ID_EVAL_NUM_TRIALS_PER_TASK = 20

# Generalist evaluation config.
GENERALIST_TASK_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_90"]
GENERALIST_NUM_TASKS_PER_SUITE = 5
GENERALIST_NUM_TRIALS_PER_TASK = 10

# OOD evaluation config.
OOD_NUM_ENVS = 5
OOD_NUM_TRIALS_PER_TASK = 10

OOD_EASY_OPTIONS = (
    " --expansion_half_len_factor 0.3 --min_distractors 1 --max_distractors 3 "
)

OOD_MEDIUM_OPTIONS = {
    "turn_on_the_stove_and_put_the_moka_pot_on_it": (
        ' --expansion_half_len_factor 0.0 --min_distractors 0 --max_distractors 0'
        ' --do_translation'
        " --translation_scales_dict '{\"moka_pot_1\": (0.0, 0.0),"
        " \"flat_stove_1\": (0.07, 0.0)}'"
        " --swap_dict '{}' "
    ),
    "put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate": (
        ' --expansion_half_len_factor 0.0 --min_distractors 0 --max_distractors 0'
        ' --do_translation'
        " --translation_scales_dict '{\"white_yellow_mug_1\": (0.0, -0.07),"
        " \"porcelain_mug_1\": (-0.05, 0.0),"
        " \"plate_1\": (0.0, 0.09),"
        " \"plate_2\": (-0.12, 0.0)}'"
        " --swap_dict '{}' "
    ),
    "put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket": (
        ' --expansion_half_len_factor 0.0 --min_distractors 0 --max_distractors 0'
        ' --do_translation'
        " --translation_scales_dict '{\"alphabet_soup_1\": (0.0, 0.0),"
        " \"cream_cheese_1\": (-0.08, 0.0),"
        " \"basket_1\": (0.03, -0.06)}'"
        " --swap_dict '{}' "
    ),
}

OOD_HARD_OPTIONS = {
    "turn_on_the_stove_and_put_the_moka_pot_on_it": [
        (
            ' --expansion_half_len_factor 0.0 --min_distractors 2 --max_distractors 2'
            ' --do_translation'
            " --translation_scales_dict '{\"moka_pot_1\": (0.0, 0.0),"
            " \"flat_stove_1\": (0.07, 0.0)}'"
            " --swap_dict '{}' "
        ),
        (
            ' --expansion_half_len_factor 0.0 --min_distractors 2 --max_distractors 2'
            ' --do_translation'
            " --translation_scales_dict '{\"moka_pot_1\": (0.0, -0.07),"
            " \"flat_stove_1\": (0.07, 0.0)}'"
            " --swap_dict '{}' "
        ),
    ],
    "put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate": [
        (
            ' --expansion_half_len_factor 0.0 --min_distractors 3 --max_distractors 3'
            ' --do_translation'
            " --translation_scales_dict '{\"white_yellow_mug_1\": (0.05, -0.09),"
            " \"porcelain_mug_1\": (-0.05, 0.0),"
            " \"plate_1\": (0.0, 0.09),"
            " \"plate_2\": (-0.12, 0.0)}'"
            " --swap_dict '{}' "
        ),
        (
            ' --expansion_half_len_factor 0.0 --min_distractors 0 --max_distractors 0'
            " --translation_scales_dict '{}'"
            " --swap_dict '{\"Living_Room_Tabletop_Manipulation\":"
            " \"Kitchen_Tabletop_Manipulation\","
            " \"living_room_table\": \"kitchen_table\"}' "
        ),
    ],
    "put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket": [
        (
            ' --expansion_half_len_factor 0.0 --min_distractors 3 --max_distractors 3'
            ' --do_translation'
            " --translation_scales_dict '{\"alphabet_soup_1\": (-0.07, 0.05),"
            " \"cream_cheese_1\": (0.04, -0.1),"
            " \"basket_1\": (0.0, -0.06)}'"
            " --swap_dict '{}' "
        ),
        (
            ' --expansion_half_len_factor 0.0 --min_distractors 3 --max_distractors 3'
            ' --do_translation'
            " --translation_scales_dict '{}'"
            " --swap_dict '{\"Living_Room_Tabletop_Manipulation\":"
            " \"Kitchen_Tabletop_Manipulation\","
            " \"living_room_table\": \"kitchen_table\"}' "
        ),
    ],
}

OOD_MULTIMODAL_OPTIONS = {
    "turn_on_the_stove_and_put_the_moka_pot_on_it": [
        (
            ' --expansion_half_len_factor 0.0 --min_distractors 0 --max_distractors 0'
            ' --do_translation'
            " --translation_scales_dict '{\"moka_pot_1\": (0.0, 0.0),"
            " \"flat_stove_1\": (0.07, 0.0)}'"
            " --swap_dict '{}' "
        ),
    ],
    "put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate": [
        (
            ' --expansion_half_len_factor 0.0 --min_distractors 3 --max_distractors 3'
            ' --do_translation'
            " --translation_scales_dict '{\"white_yellow_mug_1\": (0.05, -0.09),"
            " \"porcelain_mug_1\": (-0.05, 0.0),"
            " \"plate_1\": (0.0, 0.09),"
            " \"plate_2\": (-0.12, 0.0)}'"
            " --swap_dict '{}' "
        ),
    ],
    "put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket": [
        (
            ' --expansion_half_len_factor 0.0 --min_distractors 0 --max_distractors 0'
            " --translation_scales_dict '{}'"
            " --swap_dict '{\"Living_Room_Tabletop_Manipulation\":"
            " \"Kitchen_Tabletop_Manipulation\","
            " \"living_room_table\": \"kitchen_table\"}' "
        ),
    ],
}


def generate_all_param_combinations(
    HOST, PORT, CHECKPOINT_NAME, TASK_NAME,
    DO_ID, DO_GENERALIST, DO_OOD_EASY, DO_OOD_MEDIUM, DO_OOD_HARD, DO_OOD_MULTIMODAL,
):
    """Return a list of shell command strings for all requested eval types."""
    TASK_NAME = TASK_NAME.replace(" ", "_")
    default_args = {
        "host": HOST,
        "port": PORT,
        "results_root_dir": os.path.join(RESULTS_ROOT_DIR, CHECKPOINT_NAME),
    }
    os.makedirs(default_args["results_root_dir"], exist_ok=True)
    params_prefix = " ".join([f"--{k} {v}" for k, v in default_args.items()])

    all_param_strs = []

    # In-distribution evaluation.
    if DO_ID:
        logging.info("Generating ID evals")
        exp_name = f"ID_{CHECKPOINT_NAME}"
        cmd = (
            f"python {ID_GENERALIST_RUN_FILE} {params_prefix}"
            f" --task_suite_name {EVAL_TASK_SUITE}"
            f" --task_name '{TASK_NAME}'"
            f" --num_trials_per_task {ID_EVAL_NUM_TRIALS_PER_TASK}"
            f" --exp_name {exp_name}"
        )
        all_param_strs.append(cmd)

    # Generalist evaluation (sample tasks across suites).
    if DO_GENERALIST:
        from libero.libero import benchmark
        logging.info("Generating generalist evals")
        exp_name = f"GENERALIST_{CHECKPOINT_NAME}"

        task_set = set()
        for suite_name in GENERALIST_TASK_SUITES:
            task_suite = benchmark.get_benchmark_dict()[suite_name]()
            stride = task_suite.n_tasks // GENERALIST_NUM_TASKS_PER_SUITE
            for j in range(GENERALIST_NUM_TASKS_PER_SUITE):
                task = task_suite.get_task(j * stride)
                task_set.add((suite_name, task.language))

        logging.info(f"Number of unique generalist tasks: {len(task_set)}")
        for suite_name, tname in task_set:
            tname = tname.replace(" ", "_")
            cmd = (
                f"python {ID_GENERALIST_RUN_FILE} {params_prefix}"
                f" --task_suite_name {suite_name}"
                f" --task_name '{tname}'"
                f" --num_trials_per_task {GENERALIST_NUM_TRIALS_PER_TASK}"
                f" --exp_name {exp_name}"
            )
            all_param_strs.append(cmd)

    # OOD-easy evaluation.
    if DO_OOD_EASY:
        logging.info(f"Generating OOD-easy evals")
        exp_name = f"OOD_EASY_{CHECKPOINT_NAME}"
        for i in range(OOD_NUM_ENVS):
            cmd = (
                f"python {OOD_RUN_FILE} {params_prefix}"
                f" --task_suite_name {EVAL_TASK_SUITE}"
                f" --task_name '{TASK_NAME}'"
                f" --num_trials_per_task {OOD_NUM_TRIALS_PER_TASK}"
                f" --exp_name {exp_name}"
                f" --seed {i * 10}"
                f" {OOD_EASY_OPTIONS}"
            )
            all_param_strs.append(cmd)

    # OOD-medium evaluation.
    if DO_OOD_MEDIUM:
        ood_medium_options = OOD_MEDIUM_OPTIONS[TASK_NAME]
        logging.info(f"Generating OOD-medium evals")
        exp_name = f"OOD_MEDIUM_{CHECKPOINT_NAME}"
        for i in range(OOD_NUM_ENVS):
            cmd = (
                f"python {OOD_RUN_FILE} {params_prefix}"
                f" --task_suite_name {EVAL_TASK_SUITE}"
                f" --task_name '{TASK_NAME}'"
                f" --num_trials_per_task {OOD_NUM_TRIALS_PER_TASK}"
                f" --exp_name {exp_name}"
                f" --seed {i * 20 + 1}"
                f" {ood_medium_options}"
            )
            all_param_strs.append(cmd)

    # OOD-hard evaluation (multiple perturbation sets per task).
    if DO_OOD_HARD:
        ood_hard_options = OOD_HARD_OPTIONS[TASK_NAME]
        logging.info(f"Generating OOD-hard evals")
        for i in range(OOD_NUM_ENVS):
            for j, option in enumerate(ood_hard_options):
                exp_name = f"OOD_HARD-set-{j}_{CHECKPOINT_NAME}"
                cmd = (
                    f"python {OOD_RUN_FILE} {params_prefix}"
                    f" --task_suite_name {EVAL_TASK_SUITE}"
                    f" --task_name '{TASK_NAME}'"
                    f" --num_trials_per_task {OOD_NUM_TRIALS_PER_TASK}"
                    f" --exp_name {exp_name}"
                    f" --seed {i * 20 + 1}"
                    f" {option}"
                )
                all_param_strs.append(cmd)

    # OOD-multimodal evaluation.
    if DO_OOD_MULTIMODAL:
        ood_multimodal_options = OOD_MULTIMODAL_OPTIONS[TASK_NAME]
        logging.info(f"Generating OOD-multimodal evals")
        for i in range(OOD_NUM_ENVS):
            for j, option in enumerate(ood_multimodal_options):
                exp_name = f"OOD_MULTIMODAL-set-{j}_{CHECKPOINT_NAME}"
                cmd = (
                    f"python {OOD_RUN_FILE} {params_prefix}"
                    f" --task_suite_name {EVAL_TASK_SUITE}"
                    f" --task_name '{TASK_NAME}'"
                    f" --num_trials_per_task {OOD_NUM_TRIALS_PER_TASK}"
                    f" --exp_name {exp_name}"
                    f" --seed {i * 20 + 1}"
                    f" {option}"
                )
                all_param_strs.append(cmd)

    assert len(all_param_strs) > 0, "No jobs generated"
    return all_param_strs


def main():
    parser = argparse.ArgumentParser(
        description="Generate and print evaluation commands for LIBERO tasks.",
    )
    parser.add_argument("--host", type=str, required=True, help="Policy server host")
    parser.add_argument("--port", type=int, required=True, help="Policy server port")
    parser.add_argument("--checkpoint_name", type=str, required=True,
                        help="Checkpoint identifier (used in exp names and results dir)")
    parser.add_argument("--task_name", type=str, required=True,
                        help="LIBERO task name (spaces will be replaced with underscores)")
    parser.add_argument("--do_id", action="store_true", help="Generate in-distribution eval jobs")
    parser.add_argument("--do_generalist", action="store_true", help="Generate generalist eval jobs")
    parser.add_argument("--do_ood_easy", action="store_true", help="Generate OOD-easy eval jobs")
    parser.add_argument("--do_ood_medium", action="store_true", help="Generate OOD-medium eval jobs")
    parser.add_argument("--do_ood_hard", action="store_true", help="Generate OOD-hard eval jobs")
    parser.add_argument("--do_ood_multimodal", action="store_true",
                        help="Generate OOD-multimodal eval jobs")
    args = parser.parse_args()

    jobs = generate_all_param_combinations(
        HOST=args.host,
        PORT=args.port,
        CHECKPOINT_NAME=args.checkpoint_name,
        TASK_NAME=args.task_name,
        DO_ID=args.do_id,
        DO_GENERALIST=args.do_generalist,
        DO_OOD_EASY=args.do_ood_easy,
        DO_OOD_MEDIUM=args.do_ood_medium,
        DO_OOD_HARD=args.do_ood_hard,
        DO_OOD_MULTIMODAL=args.do_ood_multimodal,
    )

    logging.info(f"Generated {len(jobs)} job(s)")
    for job in jobs:
        print(job)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
