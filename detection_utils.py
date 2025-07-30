import pandas as pd
import json
import numpy as np
import os
import matplotlib.pyplot as plt


pretty_name = {"pval": "p-value", "bleu": "BLEU", "rouge": "ROUGE"}


def get_path(
    human: bool = False,
    data: str = "",
    model: str = "",
    temp: float = 1,
    ngram: int = 4,
    improve_id: int = 0,
    method: str = "openai",
    **kwargs,
) -> str:
    """
    Generate a path to results based on configuration parameters.

    Args:
        human: If True, returns path to human results
        data: Dataset name
        model: Model name (will be overridden by tokenizer if in kwargs)
        temp: Temperature parameter
        ngram: N-gram parameter
        improve_id: Improvement ID
        method: Method name
        **kwargs: Additional parameters, may include 'tokenizer'

    Returns:
        Path string to the results directory
    """
    with open("config.json", "r") as f:
        config = json.load(f)

    results_path = config["results_path"]
    # Override model with tokenizer if provided
    model = kwargs.get("tokenizer", model)

    base_path = f"{results_path}/{data}/{model}/{method}"

    if human:
        return f"{base_path}/human"

    return f"{base_path}/temp{temp}_ngram{ngram}/{improve_id}"


def load_results(
    json_path: str, nsamples: int = None, result_key: str = None, logging: bool = True
):
    if not os.path.exists(json_path):
        if logging:
            print(f"File not found: {json_path}")
        return []
    with open(json_path, "r") as f:
        if json_path.endswith(".json"):
            prompts = json.loads(f.read())
        else:
            prompts = [json.loads(line) for line in f.readlines()]  # load jsonl
    if result_key:
        new_prompts = [o[result_key] for o in prompts]
    else:
        new_prompts = prompts
    if nsamples is not None:
        new_prompts = new_prompts[:nsamples]
    return new_prompts


def retrieve_scores(path):
    scores_path = os.path.join(path, "scores.jsonl")
    pvals = load_results(scores_path, result_key="pvalue")
    essays_path = os.path.join(path, "results.jsonl")
    if os.path.exists(essays_path):
        essays = load_results(essays_path, result_key="result")
        return pvals, essays
    else:
        return pvals


def read_jsonl(file_path):
    """
    Reads a JSON Lines file and returns a list of dictionaries.
    """
    with open(f"{file_path}.jsonl", "r") as f:
        data = [json.loads(line) for line in f.readlines()]
    return data


def load_dataset(
    data="ETS",
    temp=0.7,
    ngram=4,
    model_list=["qwen", "phi", "llama"],
    method_list=["openai", "maryland"],
    n_samples=None,
):
    """
    Load dataset and associated metrics from model runs.

    Args:
        data: Dataset name ('ETS' or 'LOCNESS')
        temp: Temperature parameter
        ngram: N-gram parameter
        model_list: List of model names to process
        method_list: List of method names to process

    Returns:
        DataFrame containing original data and computed metrics
    """
    # Map dataset shortnames to actual filenames
    dataset_mapping = {"ETS": "ETS_corpus_sampled", "LOCNESS": "LOCNESS_sampled"}

    dataset = dataset_mapping.get(data)
    if not dataset:
        raise ValueError(
            f"Unknown dataset: {data}. Choose from {list(dataset_mapping.keys())}"
        )

    df = pd.DataFrame(read_jsonl(f"data/{dataset}"))
    # Load base dataset
    if n_samples is None:
        n_samples = len(df)

    df = df[:n_samples]

    cols = {}

    # Process all models and methods
    for model in model_list:
        for method in method_list:
            # Get human scores once
            human_scores_dir = get_path(
                human=True, data=dataset, model=model, method=method
            )
            pvals = retrieve_scores(human_scores_dir)
            df[f"Human_{method}_pval"] = pvals[:n_samples]

            # Process all improvement IDs
            for improve_id in range(1, 8):
                # Create ID string once
                id_str = f"{model}_{method}_{improve_id}"

                # Get scores path once - reuse for both scores and similarity
                scores_dir = get_path(
                    data=dataset,
                    model=model,
                    temp=temp,
                    ngram=ngram,
                    improve_id=improve_id,
                    method=method,
                )

                # Get and essays
                pvals, essays = retrieve_scores(scores_dir)
                cols[f"{id_str}_pval"] = pvals[:n_samples]
                cols[f"{id_str}_essay"] = essays[:n_samples]

                # Get similarity metrics
                similarity_path = os.path.join(scores_dir, "similarity.jsonl")
                cols[f"{id_str}_bleu"] = load_results(
                    similarity_path, result_key="bleu"
                )[:n_samples]

    # Create metrics dataframe and merge with original data
    new_metrics = pd.DataFrame(cols, index=df.index)
    return pd.concat([df, new_metrics], axis=1)


def log100(x):
    # use base 100 logarithm
    return np.log10(x + 1e-1000) / np.log10(100)  # add a small value to avoid log(0)


def get_col_name(model, method, improve_id=0, metric="pval"):
    """
    Generate a standardized column name for accessing data.

    Args:
        model: Model name; if 'Human', it will be prefixed with 'Human_'
        method: Method name
        improve_id: Improvement ID
        metric: Metric name (e.g., 'bleu', 'pval', 'score')

    Returns:
        Formatted column name string
    """
    if model == "Human":
        return f"Human_{method}_{metric}"
    return f"{model}_{method}_{improve_id}_{metric}"


def visualize_p_vals(
    pvals,
    is_outlier,
    pval_type="Marginal p-value",
    alpha=0.05,
    verbose=False,
    save_path=None,
    train_groups=None,
    title_fontsize=16,
    label_fontsize=14,
    tick_fontsize=12,
    legend_fontsize=12,
):
    """
    Visualize p-values distribution for inliers, outliers, and suspects.

    Args:
        pvals: Array of p-values
        is_outlier: Array indicating whether each sample is an inlier (0), outlier (1), or suspect (2)
        pval_type: String describing the type of p-value being plotted
        alpha: Significance level (default: 0.05)

    Returns:
        dict: Statistics for each category (false positive rate, power, suspect positive rate)
    """

    categories = [
        {"value": 0, "name": "Inliers", "stat_name": "False positive rate"},
        {"value": 1, "name": "Outliers", "stat_name": "Power"},
    ]

    # Plot the distribution of p-values of three groups
    fig, axs = plt.subplots(
        1, len(categories) + 1, figsize=(3 * (len(categories) + 1), 3)
    )
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.3)

    decision_line_label = f"Decision Boundary (α = {alpha})"
    stats = {}  # Dictionary to store statistics

    for i, category in enumerate(categories):
        # Get data for this category
        cat_pvals = pvals[is_outlier == category["value"]]
        if len(cat_pvals) == 0:
            continue

        # Set x-axis limits
        if category["value"] != 1:
            axs[i].set_xlim([0, 1])
            bins = 30
        else:
            max_val = max([np.max(cat_pvals), alpha]) + 0.01
            min_val = min([np.min(cat_pvals), alpha]) - 0.01
            axs[i].set_xlim([min_val, max_val])
            bins = 20

        axs[i].hist(cat_pvals, bins=bins)
        axs[i].set_xlabel(pval_type, fontsize=label_fontsize)
        axs[i].set_ylabel("Frequency", fontsize=label_fontsize)
        axs[i].title.set_text(category["name"])
        axs[i].title.set_fontsize(title_fontsize)
        axs[i].tick_params(axis='both', labelsize=tick_fontsize)
        # Add decision boundary line
        axs[i].axvline(x=alpha, color="red", linestyle="--", label=decision_line_label)

        # Calculate and print statistics
        stat_rate = np.mean(cat_pvals < alpha) * 100

        if verbose:
            print(f"{category['stat_name']}: {stat_rate:.2f}%")

        # Save statistics to dictionary
        stats[category["stat_name"]] = stat_rate

    # plot the histogram of all p-values, the labels are defined by the is_outlier
    # group the p-values by is_outlier, which has values 0, 1, 2
    all_pvals = []
    for i in range(3):
        all_pvals.append(pvals[is_outlier == i])
    axs[-1].hist(all_pvals, bins=30, label=["Inliers", "Outliers", "Suspects"])
    axs[-1].set_xlabel(pval_type, fontsize=label_fontsize)
    axs[-1].set_ylabel("Frequency", fontsize=label_fontsize)
    axs[-1].title.set_text("All p-values")
    axs[-1].title.set_fontsize(title_fontsize)
    axs[-1].tick_params(axis='both', labelsize=tick_fontsize)
    axs[-1].axvline(x=alpha, color="red", linestyle="--")
    axs[-1].legend(fontsize=legend_fontsize)

    # Show plot
    if verbose:
        plt.show()
    else:
        # do not show the plot
        plt.close(fig)

    if save_path:
        # Save the figure
        fig.savefig(save_path, bbox_inches="tight")

    # Return the statistics
    return stats
