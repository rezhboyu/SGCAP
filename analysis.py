"""
Analysis and visualization for the anchoring bias experiment.

Usage:
    python analysis.py                # Analyze all results
    python analysis.py --model gpt-4o-mini  # Single model
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

CONDITION_LABELS = {
    "baseline": "Baseline",
    "high_anchor": "High Anchor",
    "low_anchor": "Low Anchor",
    "counter_anchor": "Counter-Anchor",
    "sgcap": "SGCAP v1",
    "sgcap_v2": "SGCAP v2 (aware)",
}

CONDITION_COLORS = {
    "baseline": "#4C72B0",
    "high_anchor": "#DD5555",
    "low_anchor": "#55AA55",
    "counter_anchor": "#D4A017",
    "sgcap": "#8B5CF6",
    "sgcap_v2": "#E91E90",
}

ALL_CONDITIONS = ["baseline", "high_anchor", "low_anchor", "counter_anchor", "sgcap", "sgcap_v2"]


def load_results(model_filter: str | None = None) -> pd.DataFrame:
    """Load all JSONL result files into a DataFrame."""
    records = []
    for path in RESULTS_DIR.glob("*.jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

    if not records:
        print("No results found in results/ directory.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df.dropna(subset=["parsed_value"])

    if model_filter:
        df = df[df["model"] == model_filter]

    # --- Outlier filtering ---
    # 1. Remove questions with known scale/unit mismatch issues
    EXCLUDE_QUESTIONS = {
        "ptf_11",  # temperature question, model confuses with Instagram likes
        "ptf_18",  # temperature question, model confuses with view counts
        "ptf_45",  # temperature question, model confuses with large numbers
        "ptf_54",  # "In Mio. USD" - unit mismatch (model answers in millions, we store raw)
        "ptf_55",  # Reddit upvotes - model misidentifies the target number
    }
    before = len(df)
    df = df[~df["question_id"].isin(EXCLUDE_QUESTIONS)]
    excluded = before - len(df)

    # 2. Per-question outlier removal: drop responses > 10x or < 0.1x the
    #    question's baseline median (likely scale confusion on remaining questions)
    clean_rows = []
    for qid in df["question_id"].unique():
        qdf = df[df["question_id"] == qid]
        baseline_med = qdf[qdf["condition"] == "baseline"]["parsed_value"].median()
        if pd.isna(baseline_med) or baseline_med == 0:
            clean_rows.append(qdf)
            continue
        mask = (qdf["parsed_value"] <= baseline_med * 10) & (qdf["parsed_value"] >= baseline_med * 0.1)
        clean_rows.append(qdf[mask])
    df_clean = pd.concat(clean_rows, ignore_index=True)
    outlier_removed = len(df) - len(df_clean)
    df = df_clean

    print(f"Loaded {len(df)} records ({df['model'].nunique()} models, "
          f"{df['question_id'].nunique()} questions)")
    print(f"  Excluded {excluded} records from {len(EXCLUDE_QUESTIONS)} problematic questions")
    print(f"  Removed {outlier_removed} additional outlier responses (>10x or <0.1x baseline)")
    return df


def compute_anchoring_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Anchoring Index (AI) for each question and condition.
    AI = (median_anchored - median_baseline) / (anchor_value - median_baseline)

    AI close to 0 = no anchoring effect
    AI close to 1 = full anchoring (answer matches anchor)
    """
    rows = []

    for model in df["model"].unique():
        mdf = df[df["model"] == model]

        for qid in mdf["question_id"].unique():
            qdf = mdf[mdf["question_id"] == qid]
            true_val = qdf["true_value"].iloc[0]
            category = qdf["category"].iloc[0]

            baseline_median = qdf[qdf["condition"] == "baseline"]["parsed_value"].median()

            for condition in ["high_anchor", "low_anchor", "counter_anchor"]:
                cond_df = qdf[qdf["condition"] == condition]
                if cond_df.empty:
                    continue

                cond_median = cond_df["parsed_value"].median()

                if condition == "high_anchor":
                    anchor_val = qdf["high_anchor"].iloc[0] if "high_anchor" in qdf.columns else true_val * 10
                elif condition == "low_anchor":
                    anchor_val = qdf["low_anchor"].iloc[0] if "low_anchor" in qdf.columns else true_val / 10
                else:
                    anchor_val = None

                if anchor_val is not None and anchor_val != baseline_median:
                    ai = (cond_median - baseline_median) / (anchor_val - baseline_median)
                else:
                    ai = None

                rows.append({
                    "model": model,
                    "question_id": qid,
                    "category": category,
                    "condition": condition,
                    "true_value": true_val,
                    "baseline_median": baseline_median,
                    "condition_median": cond_median,
                    "anchor_value": anchor_val,
                    "anchoring_index": ai,
                })

    return pd.DataFrame(rows)


def statistical_tests(df: pd.DataFrame):
    """Run Wilcoxon signed-rank tests: baseline vs each condition per question."""
    print("\n" + "=" * 60)
    print("STATISTICAL TESTS (Wilcoxon signed-rank)")
    print("=" * 60)

    for model in df["model"].unique():
        print(f"\n--- {model} ---")
        mdf = df[df["model"] == model]

        for condition in ["high_anchor", "low_anchor", "counter_anchor", "sgcap"]:
            baseline_medians = []
            condition_medians = []

            for qid in mdf["question_id"].unique():
                qdf = mdf[mdf["question_id"] == qid]
                b = qdf[qdf["condition"] == "baseline"]["parsed_value"].median()
                c = qdf[qdf["condition"] == condition]["parsed_value"].median()
                if not np.isnan(b) and not np.isnan(c):
                    baseline_medians.append(b)
                    condition_medians.append(c)

            if len(baseline_medians) < 5:
                print(f"  {CONDITION_LABELS[condition]}: Not enough data")
                continue

            baseline_arr = np.array(baseline_medians)
            cond_arr = np.array(condition_medians)
            diffs = cond_arr - baseline_arr
            # Remove zeros (Wilcoxon can't handle them)
            nonzero = diffs != 0
            if nonzero.sum() < 5:
                print(f"  {CONDITION_LABELS[condition]}: Not enough non-zero differences")
                continue

            stat, p_value = stats.wilcoxon(diffs[nonzero])
            effect = np.median(diffs)
            print(f"  {CONDITION_LABELS[condition]:15s}: "
                  f"W={stat:.1f}, p={p_value:.6f}, "
                  f"median_diff={effect:+.2f}, "
                  f"{'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'n.s.'}")


def plot_boxplots(df: pd.DataFrame):
    """Box plots of raw estimates by condition, one subplot per model."""
    models = df["model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 6), squeeze=False)

    for i, model in enumerate(models):
        ax = axes[0, i]
        mdf = df[df["model"] == model].copy()

        # Normalize: (estimate - true_value) / true_value (percentage error)
        mdf["pct_error"] = (mdf["parsed_value"] - mdf["true_value"]) / mdf["true_value"] * 100

        # Clip extreme outliers for visualization
        mdf["pct_error"] = mdf["pct_error"].clip(-500, 500)

        order = [c for c in ALL_CONDITIONS if c in mdf["condition"].unique()]
        palette = [CONDITION_COLORS[c] for c in order]

        sns.boxplot(
            data=mdf, x="condition", y="pct_error",
            order=order, palette=palette, ax=ax,
            fliersize=2, linewidth=0.8,
        )
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([CONDITION_LABELS[c] for c in order], rotation=15)
        ax.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
        ax.set_ylabel("Percentage Error from True Value (%)")
        ax.set_xlabel("")
        ax.set_title(f"{model}")

    fig.suptitle("Distribution of Estimation Errors by Condition", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "boxplots_by_condition.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'boxplots_by_condition.png'}")


def plot_anchoring_index(ai_df: pd.DataFrame):
    """Bar chart of mean Anchoring Index by condition and model."""
    if ai_df.empty or "anchoring_index" not in ai_df.columns:
        return

    plot_df = ai_df.dropna(subset=["anchoring_index"])
    # Clip extreme AI values
    plot_df = plot_df.copy()
    plot_df["anchoring_index"] = plot_df["anchoring_index"].clip(-2, 2)

    fig, ax = plt.subplots(figsize=(8, 5))

    conditions = ["high_anchor", "low_anchor"]
    models = plot_df["model"].unique()
    x = np.arange(len(conditions))
    width = 0.35

    for i, model in enumerate(models):
        means = []
        sems = []
        for cond in conditions:
            vals = plot_df[(plot_df["model"] == model) & (plot_df["condition"] == cond)]["anchoring_index"]
            means.append(vals.mean())
            sems.append(vals.sem())
        ax.bar(x + i * width, means, width, yerr=sems, label=model, capsize=3)

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in conditions])
    ax.set_ylabel("Anchoring Index")
    ax.set_title("Mean Anchoring Index by Condition and Model")
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.5)
    ax.axhline(y=1, color="red", linestyle=":", linewidth=0.5, label="Full anchoring")
    ax.legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "anchoring_index.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'anchoring_index.png'}")


def plot_scatter_true_vs_estimate(df: pd.DataFrame):
    """Scatter plot: true value vs median estimate per condition."""
    models = df["model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 6), squeeze=False)

    for i, model in enumerate(models):
        ax = axes[0, i]
        mdf = df[df["model"] == model]

        for condition in [c for c in ALL_CONDITIONS if c in mdf["condition"].unique()]:
            medians = []
            true_vals = []
            for qid in mdf["question_id"].unique():
                qdf = mdf[(mdf["question_id"] == qid) & (mdf["condition"] == condition)]
                if not qdf.empty:
                    medians.append(qdf["parsed_value"].median())
                    true_vals.append(qdf["true_value"].iloc[0])

            ax.scatter(
                true_vals, medians,
                label=CONDITION_LABELS[condition],
                color=CONDITION_COLORS[condition],
                alpha=0.6, s=20,
            )

        # Perfect prediction line
        all_vals = mdf["true_value"].unique()
        vmin, vmax = min(all_vals), max(all_vals)
        ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=0.5, label="Perfect")
        ax.set_xscale("symlog")
        ax.set_yscale("symlog")
        ax.set_xlabel("True Value")
        ax.set_ylabel("Median Estimate")
        ax.set_title(f"{model}")
        ax.legend(fontsize=8)

    fig.suptitle("True Value vs. Median Estimate", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "scatter_true_vs_estimate.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'scatter_true_vs_estimate.png'}")


def plot_category_heatmap(ai_df: pd.DataFrame):
    """Heatmap of mean Anchoring Index by category and condition."""
    if ai_df.empty:
        return

    for model in ai_df["model"].unique():
        mdf = ai_df[(ai_df["model"] == model) & (ai_df["condition"].isin(["high_anchor", "low_anchor"]))]
        mdf = mdf.dropna(subset=["anchoring_index"])
        if mdf.empty:
            continue

        pivot = mdf.pivot_table(
            values="anchoring_index",
            index="category",
            columns="condition",
            aggfunc="mean",
        )
        pivot.columns = [CONDITION_LABELS[c] for c in pivot.columns]

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(
            pivot, annot=True, fmt=".2f", cmap="RdYlBu_r",
            center=0, ax=ax, linewidths=0.5,
        )
        ax.set_title(f"Anchoring Index by Category — {model}")
        plt.tight_layout()
        safe_model = model.replace("/", "_")
        plt.savefig(FIGURES_DIR / f"heatmap_{safe_model}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {FIGURES_DIR / f'heatmap_{safe_model}.png'}")


def plot_counter_anchor_effectiveness(df: pd.DataFrame):
    """Compare high_anchor vs counter_anchor median errors."""
    models = df["model"].unique()
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 5), squeeze=False)

    for i, model in enumerate(models):
        ax = axes[0, i]
        mdf = df[df["model"] == model]
        questions = mdf["question_id"].unique()

        high_errors = []
        counter_errors = []

        for qid in questions:
            qdf = mdf[mdf["question_id"] == qid]
            true_val = qdf["true_value"].iloc[0]

            h = qdf[qdf["condition"] == "high_anchor"]["parsed_value"].median()
            c = qdf[qdf["condition"] == "counter_anchor"]["parsed_value"].median()

            if not np.isnan(h) and not np.isnan(c):
                high_errors.append(abs(h - true_val) / abs(true_val) * 100)
                counter_errors.append(abs(c - true_val) / abs(true_val) * 100)

        ax.scatter(high_errors, counter_errors, alpha=0.5, s=20)
        max_val = max(max(high_errors, default=0), max(counter_errors, default=0))
        ax.plot([0, max_val], [0, max_val], "r--", linewidth=0.5, label="Equal error")
        ax.set_xlabel("High Anchor |% Error|")
        ax.set_ylabel("Counter-Anchor |% Error|")
        ax.set_title(f"{model}")
        ax.legend()

    fig.suptitle("Counter-Anchor Effectiveness (below line = improved)", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "counter_anchor_effectiveness.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {FIGURES_DIR / 'counter_anchor_effectiveness.png'}")


def plot_sgcap_comparison(df: pd.DataFrame):
    """Bar chart comparing MdAPE across all 5 conditions."""
    for model in df["model"].unique():
        mdf = df[df["model"] == model]
        conditions_present = [c for c in ALL_CONDITIONS if c in mdf["condition"].unique()]

        mdapes = []
        labels = []
        colors = []
        for cond in conditions_present:
            cdf = mdf[mdf["condition"] == cond]
            errors = abs(cdf["parsed_value"] - cdf["true_value"]) / abs(cdf["true_value"]) * 100
            mdapes.append(errors.median())
            labels.append(CONDITION_LABELS[cond])
            colors.append(CONDITION_COLORS[cond])

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(labels, mdapes, color=colors, edgecolor="white", linewidth=0.5)

        # Add value labels on bars
        for bar, val in zip(bars, mdapes):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

        ax.set_ylabel("Median Absolute Percentage Error (%)")
        ax.set_title(f"Estimation Accuracy by Condition — {model}")
        ax.set_ylim(0, max(mdapes) * 1.25)

        plt.tight_layout()
        safe_model = model.replace("/", "_")
        plt.savefig(FIGURES_DIR / f"sgcap_comparison_{safe_model}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {FIGURES_DIR / f'sgcap_comparison_{safe_model}.png'}")


def summary_table(df: pd.DataFrame):
    """Print a summary table of results."""
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)

    for model in df["model"].unique():
        print(f"\n--- {model} ---")
        mdf = df[df["model"] == model]

        for condition in ALL_CONDITIONS:
            cdf = mdf[mdf["condition"] == condition]
            if cdf.empty:
                continue

            # Compute median absolute percentage error
            errors = abs(cdf["parsed_value"] - cdf["true_value"]) / abs(cdf["true_value"]) * 100
            print(f"  {CONDITION_LABELS[condition]:15s}: "
                  f"n={len(cdf):5d}, "
                  f"MdAPE={errors.median():8.1f}%, "
                  f"Mean APE={errors.mean():8.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Analyze anchoring bias results")
    parser.add_argument("--model", type=str, default=None, help="Filter by model")
    args = parser.parse_args()

    df = load_results(model_filter=args.model)
    if df.empty:
        return

    # Summary stats
    summary_table(df)

    # Statistical tests
    statistical_tests(df)

    # Compute anchoring index
    ai_df = compute_anchoring_index(df)

    # Generate plots
    print("\nGenerating figures...")
    plot_boxplots(df)
    plot_anchoring_index(ai_df)
    plot_scatter_true_vs_estimate(df)
    plot_category_heatmap(ai_df)
    plot_counter_anchor_effectiveness(df)

    # SGCAP comparison
    if "sgcap" in df["condition"].unique():
        plot_sgcap_comparison(df)

    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
