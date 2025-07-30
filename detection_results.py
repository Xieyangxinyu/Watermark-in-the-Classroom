import numpy as np
import pandas as pd
import argparse

from tqdm import tqdm
from detection import *
from detection_utils import load_dataset

# reproducibility
master = np.random.SeedSequence(12345)
children = master.spawn(5)
seeds = [child.generate_state(1)[0] for child in children]

# parameters
ngram = 4
model_list = ["phi", "qwen", "llama"]
method_list = ["openai", "maryland"]
prompt_list = ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8"]


def get_args_parser():
    parser = argparse.ArgumentParser("Args", add_help=False)

    # tokenizer parameters
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="phi",
        help="tokenizer to use: phi, qwen, llama",
    )

    # watermark parameters
    parser.add_argument(
        "--method_detect",
        type=str,
        default="openai",
        help="Choose among: openai (Aaronson et al.), maryland (Kirchenbauer et al.)",
    )

    parser.add_argument(
        "--ngram",
        type=int,
        default=4,
        help="watermark context width for rng key generation",
    )

    parser.add_argument(
        "--experiment",
        type=int,
        default=3,
        help="experiment number to run: 1 (ETS), 2 (LOCNESS), 3 (Weighted)",
    )
    return parser


base_list = [1, 2, 3, 4, 5, 6]
alternatives = [[2], [3], [4], [5], [6], [7]]
adjust_alpha = 0.05


def run_conformal_tests_wrapper(
    experiment, n_trains, random_seed=42, train_groups=False, **kwargs
):
    results = []
    if experiment.run_experiment:
        for n_train in n_trains:
            df_train = experiment.df_heldout.sample(n=n_train, random_state=random_seed)
            if train_groups:
                out = experiment.run_conformal_tests(
                    df_train, train_groups=df_train["essay_id"], **kwargs
                )
            else:
                out = experiment.run_conformal_tests(df_train, **kwargs)
            if out is None:
                continue
            result = {"n_train": n_train}
            result.update(out)
            results.append(result)
    return results


args = get_args_parser().parse_args()

# prepare separate accumulators
runs = []

model = args.tokenizer
method = args.method_detect
ngram = args.ngram
experiment = args.experiment

df_ETS = load_dataset(data="ETS", model_list=[model], method_list=[method], ngram=ngram)

df_LOC = load_dataset(
    data="LOCNESS", model_list=[model], method_list=[method], ngram=ngram
)

for base in tqdm(base_list):
    for alternative in tqdm(alternatives):
        if base == min(alternative):
            continue
        if experiment == 1:
            for prompt in prompt_list:
                df_prompt = df_ETS[df_ETS["Prompt"] == prompt]
                for seed in seeds:
                    exp = WatermarkInClassroom(
                        df_prompt,
                        model,
                        method,
                        base,
                        alternative,
                        metric="pval",
                        adjust_alpha=adjust_alpha,
                        random_state=seed,
                    )
                    results = run_conformal_tests_wrapper(
                        exp, n_trains=[30, 50, 200], alpha=0.05
                    )
                    for r in results:
                        r.update(
                            {
                                "dataset": "ETS",
                                "model": model,
                                "method": method,
                                "prompt": prompt,
                                "seed": seed,
                                "base": base,
                                "alternatives": alternative,
                            }
                        )
                        runs.append(r)

        # ---- LOCNESS -----
        if experiment == 2:
            for seed in seeds:
                exp2 = WatermarkInClassroomLOCNESS(
                    df_LOC,
                    model,
                    method,
                    base,
                    alternative,
                    metric="pval",
                    adjust_alpha=adjust_alpha,
                    random_state=seed,
                )
                results = run_conformal_tests_wrapper(
                    exp2,
                    n_trains=[30, 50, 200],
                    alpha=0.05,
                    train_groups=True,
                )
                for r in results:
                    r.update(
                        {
                            "dataset": "LOCNESS",
                            "model": model,
                            "method": method,
                            "prompt": None,  # not used for LOCNESS
                            "seed": seed,
                            "base": base,
                            "alternatives": alternative,
                        }
                    )
                    runs.append(r)

        if experiment == 3:
            for seed in seeds:
                for prompt in prompt_list:
                    df_prompt = df_ETS[df_ETS["Prompt"] == prompt]
                    exp3 = WatermarkInClassroomWeighted(
                        df_prompt,
                        model,
                        method,
                        base,
                        alternative,
                        metric="pval",
                        adjust_alpha=adjust_alpha,
                        df_majority=df_LOC,
                        random_state=seed,
                    )

                    results = run_conformal_tests_wrapper(
                        exp3, n_trains=[5, 15, 30], random_seed=seed, alpha=0.05
                    )
                    for r in results:
                        r.update(
                            {
                                "dataset": "ETS",
                                "model": model,
                                "method": method,
                                "prompt": prompt,
                                "seed": seed,
                                "base": base,
                                "alternatives": alternative,
                            }
                        )

                        shared_keys = [
                            "n_train",
                            "dataset",
                            "model",
                            "method",
                            "prompt",
                            "seed",
                            "base",
                            "alternatives",
                        ]
                        shared_data = {k: r[k] for k in shared_keys}

                        categories = [
                            "In Distribution Only",
                            "Combined Unweighted",
                            "Combined Weighted (prior)",
                            "Combined Weighted (quantile)",
                            "Combined Weighted (mean)",
                        ]

                        for category in categories:
                            row = shared_data.copy()
                            row["category"] = category
                            row["False positive rate"] = float(
                                r[category]["False positive rate"]
                            )
                            if "Power" in r[category]:
                                row["Power"] = float(r[category]["Power"])
                            else:
                                row["Power"] = -1
                            runs.append(row)


if experiment == 1:
    ets_df = pd.DataFrame(runs)
    ets_cols = [
        "n_train",
        "False positive rate",
        "Power",
        "Outliers",
        "total_samples",
        "dataset",
        "model",
        "method",
        "prompt",
        "seed",
        "base",
        "alternatives",
    ]
    filename = f"results/ETS_conformal_{model}_{method}.csv"
    ets_df.to_csv(
        filename, index=False, columns=[c for c in ets_cols if c in ets_df.columns]
    )
elif experiment == 2:
    loc_df = pd.DataFrame(runs).drop(columns=["prompt"])
    loc_cols = [
        "n_train",
        "frac",
        "False positive rate",
        "Power",
        "Outliers",
        "total_samples",
        "dataset",
        "model",
        "method",
        "seed",
        "base",
        "alternatives",
    ]
    filename = f"results/LOCNESS_conformal_{model}_{method}.csv"
    loc_df.to_csv(
        filename, index=False, columns=[c for c in loc_cols if c in loc_df.columns]
    )
elif experiment == 3:
    ets_df = pd.DataFrame(runs)
    ets_cols = [
        "n_train",
        "category",
        "False positive rate",
        "Power",
        "dataset",
        "model",
        "method",
        "prompt",
        "seed",
        "base",
        "alternatives",
    ]
    filename = f"results/ETS_conformal_weighted_{model}_{method}.csv"
    ets_df.to_csv(
        filename, index=False, columns=[c for c in ets_cols if c in ets_df.columns]
    )
