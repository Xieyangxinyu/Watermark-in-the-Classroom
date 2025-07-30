import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
pd.options.mode.chained_assignment = None

from detection_utils import get_col_name, visualize_p_vals, log100, pretty_name


class WatermarkInClassroom:
    def __init__(
        self,
        df,
        model,
        method,
        base,
        alternatives,
        metric="pval",
        adjust_alpha=0.05,
        random_state=42,
    ):
        self.model = model
        self.method = method
        self.metric = metric
        self.base = base
        self.alternatives = alternatives
        self.random_state = random_state
        self.df = df
        self.split_dataset(df)
        self.adjust_outliers(adjust_alpha)
        self.num_outliers = sum(self.df_outlier["alt_suspects"] == 1)
        self.run_experiment = self.num_outliers >= max(15, adjust_alpha * len(self.df_outlier))
        self.title_fontsize = 16
        self.label_fontsize = 14
        self.legend_fontsize = 12
        self.tick_fontsize = 12

    def __repr__(self):
        return f"WatermarkInClassroom(model={self.model}, method={self.method}, metric={self.metric})"

    def adjust_outliers(self, adjust_alpha):
        """
        Adjust the outliers based on the specified alpha value.

        Args:
            alpha: Significance level for adjustment
        """
        base_bleu_col_name = get_col_name(self.model, self.method, self.base, "bleu")
        base_bleu_scores = self.df_outlier[base_bleu_col_name]
        bleu_alpha = np.percentile(self.df_outlier[base_bleu_col_name], adjust_alpha * 100)
        base_bleu_scores = base_bleu_scores.clip(upper=bleu_alpha)
        for alternative in self.alternatives:
            alternative_bleu_col_name = get_col_name(
                self.model, self.method, alternative, "bleu"
            )
            alternative_bleu_scores = self.df_outlier[alternative_bleu_col_name]
            # 1 means that the alternative is an outlier
            # 2 means that the alternative is a suspect
            with pd.option_context("mode.chained_assignment", None):
                self.df_outlier[f"suspects_{alternative}"] = (
                    alternative_bleu_scores > base_bleu_scores
                ).astype(int) + 1
        
        self.set_alt_scores(self.df_outlier, metric=self.metric)


    def split_dataset(self, df):
        self.df_heldout = df.sample(n=200, random_state=self.random_state)
        self.df_outlier = df.drop(self.df_heldout.index)

    def set_alt_indices(self, df):
        if len(self.alternatives) > 1:
            rng = np.random.RandomState(self.random_state)
            # assign alt_index and alt_score in one go
            df["alt_index"] = rng.choice(self.alternatives, size=len(df))
        else:
            # repeat the single alternative for all rows
            df.loc[:, "alt_index"] = self.alternatives[0]

    def set_alt_scores(self, df, metric="pval"):
        """
        Get alternative scores based on the alt_index column.

        Args:
            df: DataFrame containing the data
            metric: Metric name to retrieve

        Returns:
            NumPy array of scores
        """

        if len(self.alternatives) == 1:
            df["alt_score"] = df[
                get_col_name(self.model, self.method, self.alternatives[0], metric)
            ]
            df["alt_suspects"] = df[f"suspects_{self.alternatives[0]}"]
            self.set_alt_indices(df)
        else:
            if "alt_index" not in df.columns:
                self.set_alt_indices(df)

            # Create a function to access the specific column for each row
            def get_score(row, idx):
                return row[get_col_name(self.model, self.method, idx, metric)]

            # Apply the function
            df["alt_score"] = np.array(
                [
                    get_score(row, idx)
                    for (_, row), idx in zip(df.iterrows(), df["alt_index"])
                ]
            )

            df["alt_suspects"] = np.array(
                [
                    row[f"suspects_{idx}"]
                    for (_, row), idx in zip(df.iterrows(), df["alt_index"])
                ]
            )

    def visualize_outlier(
        self,
        df_inlier,
        df_outlier,
        metric="pval",
        transform=lambda x: x,
        mode="show",
        alpha=0.05,
    ):
        """
        Visualize the outliers in the dataset.

        Args:
            df_inlier: DataFrame containing inliers
            df_outlier: DataFrame containing outliers
            metric: Metric name to visualize
            transform: Transformation function for the scores
            mode: Mode of visualization ('show', 'hide', "suspect")
            alpha: Alpha value for decision boundary
        """

        # Set up the plot
        fig, ax = plt.subplots(figsize=(6, 4.5)) # Increased figure size

        if metric == "pval":
            transform = log100

        base_col_name = get_col_name(self.model, self.method, self.base, metric)

        # Extract and transform scores
        base_scores = transform(df_inlier[base_col_name])

        self.set_alt_scores(df_outlier, metric=metric)

        alt_scores = transform(df_outlier["alt_score"])

        # Combine for shared binning
        scores = [base_scores, alt_scores]

        bins = np.linspace(
            np.concatenate(scores).min(), np.concatenate(scores).max(), 31
        )

        if mode == "show":
            color = ["blue", "orange"]
            label = ["Following Guideline", "Not Following Guideline"]
        elif mode == "hide":
            color = ["grey", "grey"]
            label = []
        else: # mode == "suspect"
            color = ["blue", "purple", "orange"]
            label = ["Following Guideline", "Suspect", "Outliers"]
            # Ensure 'alt_suspects' column exists in df_outlier for these modes
            outlier_scores = alt_scores[df_outlier["alt_suspects"] == 1]
            suspect_scores = alt_scores[df_outlier["alt_suspects"] == 2]
            scores = [base_scores, suspect_scores, outlier_scores]

        # Stack the histograms
        ax.hist(scores, bins=bins, stacked=True, label=label, color=color, alpha=0.6)

        # draw a vertical line for the decision boundary
        if mode != "suspect":
            if metric == "pval":
                ax.axvline(
                    x=transform(alpha),
                    color="red",
                    linestyle="--",
                    label=f"Default Decision Boundary, α = {alpha}",
                    linewidth=2 # Increased line width for emphasis
                )
            else:
                # use the 5th percentile of the base scores as the decision boundary
                heldout_base_scores = self.df_heldout[base_col_name]
                ax.axvline(
                    x=transform(np.percentile(heldout_base_scores, alpha * 100)),
                    color="red",
                    linestyle="--",
                    label=f"Decision Boundary, α = {alpha}",
                    linewidth=2 # Increased line width for emphasis
                )

        # Plot formatting
        ax.set_xlabel(f"{pretty_name[metric]}", fontsize=self.label_fontsize)
        ax.set_ylabel("Frequency", fontsize=self.label_fontsize)
        if mode != "hide":
            ax.legend(fontsize=self.legend_fontsize)

        # Set tick label font sizes
        ax.tick_params(axis='x', labelsize=self.tick_fontsize)
        ax.tick_params(axis='y', labelsize=self.tick_fontsize)


        if metric in ["bleu", "rouge", "rep"]:
            # identify a high score essay
            high_score_prompt = df_outlier.iloc[np.argmax(alt_scores)]
            index = high_score_prompt["alt_index"]
            alt_col_name = get_col_name(self.model, self.method, index, metric)
            print(
                f"High Similarity Score: {metric}\n{high_score_prompt[alt_col_name]}, Prompt type: {index}"
            )
            print(f"Original Essay:\n{high_score_prompt['Human']}")
            alt_col_name = get_col_name(self.model, self.method, index, "essay")
            print(f"Editted Essay:\n{high_score_prompt[alt_col_name]}")

        plt.show()
        # save the figure
        fig.savefig(
            f"figs/{self.model}_{self.method}_{self.base}_{mode}_{metric}.png",
            dpi=300,
            bbox_inches="tight",
        )

    @staticmethod
    def get_marginal_pvals(scores_train, scores_test, train_groups=None, save_path=None):
        # Apply the marginal conformal p-values to reject the null hypothesis

        scores_train_mat = np.tile(scores_train, (len(scores_test), 1))
        pvals_numerator = np.sum(
            scores_train_mat <= scores_test.reshape(len(scores_test), 1), 1
        )
        pvals_marginal = (1.0 + pvals_numerator) / (1.0 + len(scores_train))

        return pvals_marginal

    def run_conformal_tests(
        self,
        df_train,
        alpha=0.05,
        verbose=False,
        **kwargs,
    ):
        
        if self.run_experiment == False:
            print(
                f"Skipping experiment for {self.model}, {self.method}, {self.base} due to insufficient outliers (< 30)."
            )
            return None

        base_col_name = get_col_name(self.model, self.method, self.base, self.metric)
        scores_train = df_train[base_col_name].to_numpy()
        df_test = self.df_outlier

        scores_test = np.concatenate(
            [
                df_test[base_col_name].to_numpy(),
                df_test["alt_score"].to_numpy(),
            ]
        )
        is_outlier = np.concatenate(
            [np.zeros(len(df_test)), df_test["alt_suspects"]]
        )

        if verbose:
            print(f"Number of inliers: {sum(is_outlier == 0)}")
            print(f"Number of outliers: {sum(is_outlier == 1)}")
            print("Marginal Conformal p-values")

        results = {}

        pvals_marginal = self.get_marginal_pvals(scores_train, scores_test, **kwargs)

        results |= visualize_p_vals(
            pvals_marginal,
            is_outlier,
            pval_type="Marginal p-value",
            alpha=alpha,
            verbose=verbose,
            **kwargs
        )
        results["Outliers"] = sum(is_outlier == 1)
        results["total_samples"] = len(df_test)

        return results

    def visualize_base_against_human(self, transform=log100, alpha=0.05):
        human_col_name = get_col_name("Human", self.method, metric=self.metric)
        human_scores = transform(self.df_heldout[human_col_name])
        base_col_name = get_col_name(self.model, self.method, self.base, self.metric)
        base_scores = transform(self.df_heldout[base_col_name])
        all_scores = np.concatenate([human_scores, base_scores])
        bins = np.linspace(all_scores.min(), all_scores.max(), 31)

        fig, ax = plt.subplots(figsize=(6, 4.5))

        ax.hist(human_scores, bins=bins, alpha=0.5, label=f"Human", color="grey")
        # add 95% upper bound
        ax.axvline(
            x=transform(alpha),
            color="grey",
            linestyle="--",
            label=f"Default Decision Boundary, α = {alpha}",
        )
        base_label = "Grammar Checker" if self.base == 0 else f"Prompt {self.base}"
        ax.hist(base_scores, bins=bins, alpha=0.5, label=base_label, color="blue")

        ax.set_xlabel("p-value", fontsize=self.label_fontsize)
        ax.set_ylabel("Frequency", fontsize=self.label_fontsize)
        ax.set_title("Distribution of Scores: Human vs. Base Model", fontsize=self.title_fontsize)
        ax.legend(fontsize=self.legend_fontsize)

        # Set tick label font sizes
        ax.tick_params(axis='x', labelsize=self.tick_fontsize)
        ax.tick_params(axis='y', labelsize=self.tick_fontsize)

        plt.show()
        # save the figure
        fig.savefig(
            f"figs/{self.model}_{self.method}_{base_label}_human.png",
            dpi=300,
            bbox_inches="tight",
        )

        human_below_5 = (
            len(self.df_heldout[human_scores < transform(alpha)])
            / len(self.df_heldout)
            * 100
        )
        print(f"Human, {human_below_5:.2f} % samples below α = {alpha}")
        base_below_5 = (
            len(self.df_heldout[base_scores < transform(alpha)])
            / len(self.df_heldout)
            * 100
        )
        print(f"{base_label}, {base_below_5:.2f} % samples below α = {alpha}")


class WatermarkInClassroomLOCNESS(WatermarkInClassroom):
    def __init__(
        self,
        df,
        model,
        method,
        base,
        alternatives,
        metric="pval",
        adjust_alpha=0.05,
        random_state=42,
    ):
        super().__init__(
            df, model, method, base, alternatives, metric, adjust_alpha, random_state
        )

    @staticmethod
    def get_marginal_pvals(scores_train, scores_test, train_groups=None, save_path=None):
        # HCP
        # The conformal p-values are computed in the following way:
        # $$\frac{1}{(K+1)}[1 + \sum_{k=1}^{K}\frac{\sum_{i=1}^{n_k} {\bf 1}\{s(X_i) \le s(x)\}}{n_k}]$$

        assert train_groups is not None, "train_groups must be provided for LOCNESS."
        
        group_labels = np.unique(train_groups)
        K = len(group_labels)
        pvals = np.zeros(len(scores_test))

        for i in group_labels:
            # get the scores for the group
            scores_train_group = scores_train[train_groups == i]
            n_k = len(scores_train_group)
            # compute the p-values
            pvals_numerator = np.sum(
                scores_train_group <= scores_test.reshape(len(scores_test), 1), 1
            )
            pvals += (pvals_numerator) / (n_k)

        pvals = (1.0 + pvals) / (K + 1)
        return pvals


class WatermarkInClassroomWeighted(WatermarkInClassroom):
    def __init__(
        self,
        df,
        model,
        method,
        base,
        alternatives,
        metric="pval",
        adjust_alpha=0.05,
        random_state=42,
        df_majority=None,
    ):
        super().__init__(
            df, model, method, base, alternatives, metric, adjust_alpha, random_state
        )
        self.df_majority = df_majority

    def compare_score_distribution(self, metric, transform=lambda x: x):

        assert self.df_majority is not None, "Majority dataset is not provided."

        if metric == "pval":
            transform = log100

        base_col_name = get_col_name(self.model, self.method, self.base, metric)

        model_scores1 = transform(self.df_heldout[base_col_name])
        model_scores2 = transform(self.df_majority[base_col_name])

        all_scores = np.concatenate([model_scores1, model_scores2])
        bins = np.linspace(all_scores.min(), all_scores.max(), 101)  # 100 bins
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(model_scores1, bins=bins, alpha=0.5, label=f"ETS", color="blue")
        ax.hist(model_scores2, bins=bins, alpha=0.5, label=f"LOCNESS", color="green")
        ax.axvline(x=np.quantile(model_scores1, 0.05), color="blue", linestyle="--")
        ax.axvline(x=np.quantile(model_scores2, 0.05), color="green", linestyle="--")
        # Use class attributes for font sizes if available, else defaults
        ax.set_xlabel(f"{metric} score")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Score Distribution: ETS vs LOCNESS")
        ax.legend()
        ax.tick_params(axis='x')
        ax.tick_params(axis='y')
        plt.show()
        fig.savefig(
            f"figs/compare_{self.model}_{self.method}_{self.base}_{metric}.png",
            bbox_inches="tight",
            dpi=300,
        )

    def estimate_density_ratio(self, p_scores, q_scores, verbose=True, alpha=0.05):

        """
        Estimate the density ratio between two sets of scores using kernel density estimation.
        Args:
            p_scores: Scores from the majority class (p)
            q_scores: Scores from the minority class (q)
            verbose: Whether to plot the density ratio
        """

        def get_density_q(q_scores, mu_q, mu_p, q_std, p_std, kde_p):
            if mu_q > mu_p:
                mu_q = mu_p
                q_std = p_std
            if len(q_scores) <= 5:
                return lambda x: kde_p(
                    x - mu_q + mu_p
                )
            else:
                return lambda x: kde_p(
                    (x - mu_q) / q_std * p_std + mu_p
                )

        kde_p = gaussian_kde(
            p_scores, bw_method=0.5
        )

        if len(q_scores) <= 0.5/alpha:
            quantile_q = np.min(q_scores)
            quantile_p = np.quantile(p_scores, 1/(len(q_scores)))
        elif len(q_scores) <= 1/alpha:
            quantile_q = np.quantile(q_scores, alpha * 2)
            quantile_p = np.quantile(p_scores, alpha * 2)
        else:
            quantile_q = np.quantile(q_scores, alpha)
            quantile_p = np.quantile(p_scores, alpha)
        
        mu_q = q_scores.mean()
        mu_p = p_scores.mean()

        q_std = q_scores.std()
        p_std = p_scores.std()

        density_q_quantile = get_density_q(
            q_scores, quantile_q, quantile_p, q_std, p_std, kde_p
        )
        density_q_mean = get_density_q(
            q_scores, mu_q, mu_p, q_std, p_std, kde_p
        )
        
        if verbose:
            plt.figure(figsize=(5, 3))
            x = np.linspace(min(p_scores), max(p_scores), 1000)
            plt.plot(x, kde_p(x), label="p_scores")
            plt.plot(x, density_q_quantile(x), linestyle="--", label="q_scores (quantile method)")
            plt.plot(x, density_q_mean(x), linestyle=":", label="q_scores (mean method)")
            plt.xlabel("Score")
            plt.ylabel("Density")
            plt.legend()
            plt.title("Densities")
            plt.show()

        return kde_p, density_q_quantile, density_q_mean

    def calc_weighted_score(self, y, w_y, alpha, train_scores, train_weights):

        weight_total = np.sum(train_weights) + w_y
        pi = train_weights / weight_total

        train_scores = np.array(train_scores)
        sorted_indices = np.argsort(train_scores)
        sorted_R = train_scores[sorted_indices]
        sorted_pi = pi[sorted_indices]

        cumulative = np.cumsum(sorted_pi) + w_y / weight_total
        idx = np.searchsorted(cumulative, alpha)
        idx = min(idx, len(sorted_R) - 1)
        quantile_threshold = sorted_R[idx]

        return y < quantile_threshold

    def run_conformal_tests(
        self, df_train, alpha=0.05, verbose=False, **kwargs
    ):

        if self.run_experiment == False:
            print(
                f"Skipping experiment for {self.model}, {self.method}, {self.base} due to insufficient outliers (< 30)."
            )
            return None

        assert self.df_majority is not None, "Majority dataset is not provided."

        base_col_name = get_col_name(self.model, self.method, self.base, self.metric)
        transform = log100 if self.metric == "pval" else lambda x: x
        q_scores = df_train[base_col_name].to_numpy()
        q_scores = transform(q_scores)
        train_scores = self.df_majority[base_col_name].to_numpy()
        train_scores = transform(train_scores)
        train_scores = np.concatenate([train_scores, q_scores])
        
        kde_p, density_q_quantile, density_q_mean = self.estimate_density_ratio(
            p_scores=train_scores,
            q_scores=q_scores,
            verbose=verbose,
            alpha=alpha,
        )

        df_test = self.df_outlier

        train_scores = train_scores

        self.set_alt_scores(df_test, metric=self.metric)

        scores_test = np.concatenate(
            [
                df_test[base_col_name].to_numpy(),
                df_test["alt_score"].to_numpy(),
            ]
        )
        scores_test = transform(scores_test)

        is_outlier = np.concatenate(
            [np.zeros(len(df_test)), df_test["alt_suspects"]]
        )

        if verbose:
            print(f"Number of outliers: {sum(is_outlier == 1)}")
            print("Marginal Conformal p-values")

        results = {
            "In Distribution Only": {},
            "Combined Unweighted": {},
            "Combined Weighted (quantile)": {},
            "Combined Weighted (mean)": {},

        }

        p_vals_indistribution = self.get_marginal_pvals(q_scores, scores_test, **kwargs)
        results["In Distribution Only"] = visualize_p_vals(
            p_vals_indistribution,
            is_outlier,
            pval_type="Marginal p-value",
            alpha=alpha,
            verbose=False,
        )

        p_vals_combined = self.get_marginal_pvals(train_scores, scores_test, **kwargs)
        results["Combined Unweighted"] = visualize_p_vals(
            p_vals_combined,
            is_outlier,
            pval_type="Marginal p-value",
            alpha=alpha,
            verbose=False,
        )

        def compute_weight(score, option):
            option_dict = {
                "quantile": (kde_p, density_q_quantile),
                "mean": (kde_p, density_q_mean),
            }
            density_p, density_q = option_dict[option]
            weight = density_q(score) / density_p(score)
            return weight

        for option in ["quantile", "mean"]:
            train_weights = compute_weight(
                train_scores, option
            )
            decision = np.array(
                [
                    self.calc_weighted_score(
                        y, compute_weight(y, option), alpha, train_scores, train_weights
                    ) if y >= min(train_scores) else True for y in scores_test
                ]
            )

            results[f"Combined Weighted ({option})"]["False positive rate"] = (
                np.mean(decision[is_outlier == 0] == 1) * 100
            )
            if len(decision[is_outlier == 1]) > 0:
                results[f"Combined Weighted ({option})"]["Power"] = (
                    np.mean(decision[is_outlier == 1] == 1) * 100
                )
            else:
                results[f"Combined Weighted ({option})"]["Power"] = -1

        if verbose:
            for method in results.keys():
                print(f"{method} False positive rate: {results[method]['False positive rate']:.2f}")
                print(f"{method} Power: {results[method]['Power']:.2f}")
        
        return results
