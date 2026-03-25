"""
Generate publication-quality figures for the anchoring bias paper.

Usage:
    python paper_figures.py
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ── Config ──────────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Exclude problematic questions (scale/unit mismatch)
EXCLUDE_QUESTIONS = {"ptf_11", "ptf_18", "ptf_45", "ptf_54", "ptf_55"}

# Remap PTF event names → abstract task domains
DOMAIN_MAP = {
    # Stock / Financial
    "Apple WWDC": "Stock & Finance",
    "U.S. Open Golf": "Stock & Finance",
    "Sotheby's auction I": "Stock & Finance",
    "Auction: Sotheby's Contemporary Art Day Auction ": "Stock & Finance",
    "World Cup Other": "Stock & Finance",
    "FORMULA 1 Pirelli Grand Prix De France 2018": "Stock & Finance",
    "Elon Musk Birthday": "Stock & Finance",
    "Fourth of July": "Stock & Finance",
    "On this day": "Stock & Finance",
    "Montreux Jazz Festival": "Stock & Finance",
    "FB App anniversary": "Stock & Finance",
    "Twitter birthday": "Stock & Finance",
    "Festival d'été de Québec": "Stock & Finance",
    "MoMA photography": "Stock & Finance",
    # Temperature
    "Little Big Town concert": "Temperature",
    "Ottawa Craft and Beer Festival": "Temperature",
    "Big Five Marathon": "Temperature",
    "Ed Sheeran concert": "Temperature",
    "Wimbledon": "Temperature",
    "Tour de France": "Temperature",
    "World Cup final": "Temperature",
    "Taylor Swift in London": "Temperature",
    # Social Media
    "Sugarland - new music video": "Social Media",
    "Tim McGraw & Faith Hill Soul2Soul Tour": "Social Media",
    "Montreal en Arts": "Social Media",
    "Friday the 13th": "Social Media",
    # Counting / Enumeration
    "Royal Ascot": "Counting",
    "City of Vancouver": "Counting",
    "Celine Dion World Tour": "Counting",
    "Canada Day": "Counting",
    "Paris Fashion Week": "Counting",
    "BBC Proms": "Counting",
    "Box Office Mojo": "Counting",
    # Sports Stats
    "World Cup: Brazil vs. Belgium": "Sports Stats",
    "Croatia vs. England": "Sports Stats",
    # Flights
    "Flight statistics": "Flights",
    "Flight cancellations": "Flights",
    # Film
    "Film release": "Film Revenue",
}

CONDITION_ORDER = ["baseline", "high_anchor", "low_anchor", "counter_anchor"]
CONDITION_LABELS = {
    "baseline": "Baseline",
    "high_anchor": "High Anchor",
    "low_anchor": "Low Anchor",
    "counter_anchor": "Counter-Anchor",
}
CONDITION_COLORS = {
    "baseline": "#4C72B0",
    "high_anchor": "#C44E52",
    "low_anchor": "#55A868",
    "counter_anchor": "#CCB974",
}

# ── Style ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ── Data Loading ────────────────────────────────────────────────────────
def load_clean_data() -> pd.DataFrame:
    records = []
    for p in RESULTS_DIR.glob("*.jsonl"):
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    df = pd.DataFrame(records).dropna(subset=["parsed_value"])
    df = df[~df["question_id"].isin(EXCLUDE_QUESTIONS)]

    # Per-question outlier removal
    clean = []
    for qid in df["question_id"].unique():
        qdf = df[df["question_id"] == qid]
        bm = qdf[qdf["condition"] == "baseline"]["parsed_value"].median()
        if pd.isna(bm) or bm == 0:
            clean.append(qdf)
            continue
        mask = (qdf["parsed_value"] <= bm * 10) & (qdf["parsed_value"] >= bm * 0.1)
        clean.append(qdf[mask])
    df = pd.concat(clean, ignore_index=True)

    # Add domain mapping
    df["domain"] = df["category"].map(DOMAIN_MAP).fillna("Other")
    return df


def compute_ai(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Anchoring Index per question per model."""
    rows = []
    for model in df["model"].unique():
        mdf = df[df["model"] == model]
        for qid in mdf["question_id"].unique():
            qdf = mdf[mdf["question_id"] == qid]
            bm = qdf[qdf["condition"] == "baseline"]["parsed_value"].median()
            domain = qdf["domain"].iloc[0]
            tv = qdf["true_value"].iloc[0]
            for cond in ["high_anchor", "low_anchor"]:
                cdf = qdf[qdf["condition"] == cond]
                if cdf.empty:
                    continue
                cm = cdf["parsed_value"].median()
                if cond == "high_anchor":
                    av = qdf["high_anchor"].iloc[0] if "high_anchor" in qdf.columns else tv * 10
                else:
                    av = qdf["low_anchor"].iloc[0] if "low_anchor" in qdf.columns else tv / 10
                if av is not None and av != bm:
                    ai = (cm - bm) / (av - bm)
                else:
                    ai = None
                rows.append({
                    "model": model, "question_id": qid, "domain": domain,
                    "condition": cond, "anchoring_index": ai,
                    "true_value": tv, "baseline_median": bm, "condition_median": cm,
                })
    return pd.DataFrame(rows)


# ── Figure 1: Box plots ────────────────────────────────────────────────
def fig1_boxplots(df):
    models = sorted(df["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 4.5),
                             squeeze=False, sharey=True)
    for i, model in enumerate(models):
        ax = axes[0, i]
        mdf = df[df["model"] == model].copy()
        mdf["pct_error"] = (mdf["parsed_value"] - mdf["true_value"]) / mdf["true_value"].abs() * 100
        mdf["pct_error"] = mdf["pct_error"].clip(-200, 200)

        palette = [CONDITION_COLORS[c] for c in CONDITION_ORDER]
        sns.boxplot(
            data=mdf, x="condition", y="pct_error", hue="condition",
            order=CONDITION_ORDER, hue_order=CONDITION_ORDER,
            palette=palette, ax=ax, fliersize=1.5, linewidth=0.7,
            legend=False, width=0.65,
        )
        ax.set_xticks(range(len(CONDITION_ORDER)))
        ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER],
                           rotation=20, ha="right")
        ax.axhline(0, color="grey", ls="--", lw=0.5)
        ax.set_ylabel("Estimation Error (%)" if i == 0 else "")
        ax.set_xlabel("")
        ax.set_title(model, fontweight="bold")

    fig.suptitle("Figure 1. Distribution of Estimation Errors by Condition",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_boxplots.png")
    plt.close()
    print("[OK] fig1_boxplots.png")


# ── Figure 2: Anchoring Index bar chart ────────────────────────────────
def fig2_anchoring_index(ai_df):
    plot_df = ai_df.dropna(subset=["anchoring_index"]).copy()
    plot_df["anchoring_index"] = plot_df["anchoring_index"].clip(-2, 2)

    models = sorted(plot_df["model"].unique())
    conds = ["high_anchor", "low_anchor"]
    x = np.arange(len(conds))
    width = 0.3
    colors = ["#C44E52", "#55A868"]

    fig, ax = plt.subplots(figsize=(5, 4))
    for i, model in enumerate(models):
        means, sems = [], []
        for c in conds:
            v = plot_df[(plot_df["model"] == model) & (plot_df["condition"] == c)]["anchoring_index"]
            means.append(v.mean())
            sems.append(v.sem())
        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, means, width, yerr=sems, capsize=3,
                      color=colors, edgecolor="white", linewidth=0.5,
                      label=model if i == 0 else None)
        # Value labels
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                    f"{m:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in conds])
    ax.set_ylabel("Anchoring Index (AI)")
    ax.axhline(0, color="grey", ls="--", lw=0.5)
    ax.axhline(1, color="red", ls=":", lw=0.5, alpha=0.5)
    ax.text(1.02, 1.0, "Full\nanchoring", transform=ax.get_yaxis_transform(),
            fontsize=8, color="red", alpha=0.6, va="center")
    ax.set_ylim(-0.1, 1.15)
    if len(models) > 1:
        ax.legend(loc="upper left", frameon=False)

    fig.suptitle("Figure 2. Mean Anchoring Index by Condition",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_anchoring_index.png")
    plt.close()
    print("[OK] fig2_anchoring_index.png")


# ── Figure 3: Heatmap by domain ───────────────────────────────────────
def fig3_heatmap(ai_df):
    for model in sorted(ai_df["model"].unique()):
        mdf = ai_df[(ai_df["model"] == model)].dropna(subset=["anchoring_index"]).copy()
        mdf["anchoring_index"] = mdf["anchoring_index"].clip(-2, 2)
        if mdf.empty:
            continue

        pivot = mdf.pivot_table(
            values="anchoring_index", index="domain", columns="condition",
            aggfunc="mean",
        )
        pivot = pivot.reindex(columns=["high_anchor", "low_anchor"])
        pivot.columns = ["High Anchor", "Low Anchor"]
        pivot = pivot.sort_values("High Anchor", ascending=False)

        fig, ax = plt.subplots(figsize=(5, 0.5 * len(pivot) + 1.5))
        sns.heatmap(
            pivot, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0,
            ax=ax, linewidths=0.8, linecolor="white",
            cbar_kws={"label": "Anchoring Index", "shrink": 0.8},
            vmin=-0.5, vmax=1.5,
        )
        ax.set_ylabel("")
        ax.set_xlabel("")
        safe = model.replace("/", "_")
        fig.suptitle(f"Figure 3. Anchoring Index by Task Domain — {model}",
                     fontsize=12, y=1.02)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"fig3_heatmap_{safe}.png")
        plt.close()
        print(f"[OK] fig3_heatmap_{safe}.png")


# ── Figure 4: Counter-anchor effectiveness scatter ─────────────────────
def fig4_counter_effectiveness(df):
    models = sorted(df["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 5),
                             squeeze=False)
    for i, model in enumerate(models):
        ax = axes[0, i]
        mdf = df[df["model"] == model]

        high_err, counter_err, labels = [], [], []
        for qid in mdf["question_id"].unique():
            qdf = mdf[mdf["question_id"] == qid]
            tv = qdf["true_value"].iloc[0]
            h = qdf[qdf["condition"] == "high_anchor"]["parsed_value"].median()
            c = qdf[qdf["condition"] == "counter_anchor"]["parsed_value"].median()
            if pd.notna(h) and pd.notna(c) and tv != 0:
                he = abs(h - tv) / abs(tv) * 100
                ce = abs(c - tv) / abs(tv) * 100
                high_err.append(he)
                counter_err.append(ce)

        ax.scatter(high_err, counter_err, alpha=0.55, s=25, c="#4C72B0",
                   edgecolors="white", linewidth=0.3)
        mx = max(max(high_err, default=1), max(counter_err, default=1)) * 1.05
        ax.plot([0, mx], [0, mx], "r--", lw=0.8, alpha=0.6)
        ax.fill_between([0, mx], [0, 0], [0, mx], alpha=0.04, color="green")
        ax.fill_between([0, mx], [0, mx], [mx, mx], alpha=0.04, color="red")
        ax.text(mx * 0.7, mx * 0.15, "Improved", fontsize=9, color="green", alpha=0.6)
        ax.text(mx * 0.15, mx * 0.85, "Worsened", fontsize=9, color="red", alpha=0.6)
        ax.set_xlabel("High Anchor |Error| (%)")
        ax.set_ylabel("Counter-Anchor |Error| (%)" if i == 0 else "")
        ax.set_title(model, fontweight="bold")
        ax.set_xlim(0, mx)
        ax.set_ylim(0, mx)

    fig.suptitle("Figure 4. Counter-Anchor Effectiveness",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig4_counter_effectiveness.png")
    plt.close()
    print("[OK] fig4_counter_effectiveness.png")


# ── Figure 5: True vs Estimate scatter ─────────────────────────────────
def fig5_scatter(df):
    models = sorted(df["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(5.5 * len(models), 5),
                             squeeze=False)
    for i, model in enumerate(models):
        ax = axes[0, i]
        mdf = df[df["model"] == model]
        for cond in CONDITION_ORDER:
            medians, trues = [], []
            for qid in mdf["question_id"].unique():
                qdf = mdf[(mdf["question_id"] == qid) & (mdf["condition"] == cond)]
                if not qdf.empty:
                    medians.append(qdf["parsed_value"].median())
                    trues.append(qdf["true_value"].iloc[0])
            ax.scatter(trues, medians, label=CONDITION_LABELS[cond],
                       color=CONDITION_COLORS[cond], alpha=0.55, s=18,
                       edgecolors="white", linewidth=0.3)

        vals = mdf["true_value"].unique()
        lo, hi = min(vals), max(vals)
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.5, alpha=0.4)
        ax.set_xscale("symlog")
        ax.set_yscale("symlog")
        ax.set_xlabel("True Value")
        ax.set_ylabel("Median Estimate" if i == 0 else "")
        ax.set_title(model, fontweight="bold")
        ax.legend(fontsize=8, frameon=False, loc="upper left")

    fig.suptitle("Figure 5. True Value vs. Median Estimate (log scale)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig5_scatter.png")
    plt.close()
    print("[OK] fig5_scatter.png")


# ── Figure 6: Question-level anchoring direction summary ───────────────
def fig6_direction_summary(df):
    models = sorted(df["model"].unique())
    fig, axes = plt.subplots(1, len(models), figsize=(4.5 * len(models), 4),
                             squeeze=False)

    for idx, model in enumerate(models):
        ax = axes[0, idx]
        mdf = df[df["model"] == model]
        both, high_only, low_only, none_ = 0, 0, 0, 0

        for qid in mdf["question_id"].unique():
            qdf = mdf[mdf["question_id"] == qid]
            bm = qdf[qdf["condition"] == "baseline"]["parsed_value"].median()
            hm = qdf[qdf["condition"] == "high_anchor"]["parsed_value"].median()
            lm = qdf[qdf["condition"] == "low_anchor"]["parsed_value"].median()
            ha = qdf["high_anchor"].iloc[0] if "high_anchor" in qdf.columns else None
            la = qdf["low_anchor"].iloc[0] if "low_anchor" in qdf.columns else None

            h_pull = (ha is not None) and ((ha > bm and hm > bm) or (ha < bm and hm < bm)) and hm != bm
            l_pull = (la is not None) and ((la < bm and lm < bm) or (la > bm and lm > bm)) and lm != bm

            if h_pull and l_pull:
                both += 1
            elif h_pull:
                high_only += 1
            elif l_pull:
                low_only += 1
            else:
                none_ += 1

        total = both + high_only + low_only + none_
        categories = ["Both\ndirections", "High\nonly", "Low\nonly", "No\neffect"]
        values = [both, high_only, low_only, none_]
        colors = ["#C44E52", "#E8866A", "#7FBC8C", "#B0B0B0"]

        bars = ax.bar(categories, values, color=colors, edgecolor="white", width=0.6)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{v}\n({v/total*100:.0f}%)", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Number of Questions")
        ax.set_title(model, fontweight="bold")
        ax.set_ylim(0, max(values) * 1.35)

    fig.suptitle("Figure 6. Anchoring Effect Direction per Question",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fig6_direction_summary.png")
    plt.close()
    print("[OK] fig6_direction_summary.png")


# ── Table: Summary statistics (printed) ────────────────────────────────
def print_summary_table(df, ai_df):
    print("\n" + "=" * 70)
    print("TABLE 1: Summary Statistics")
    print("=" * 70)
    for model in sorted(df["model"].unique()):
        mdf = df[df["model"] == model]
        print(f"\n{'Condition':18s} {'N':>6s} {'MdAPE':>8s} {'MeanAPE':>9s} {'AI±SE':>12s}")
        print("-" * 55)
        for cond in CONDITION_ORDER:
            cdf = mdf[mdf["condition"] == cond]
            if cdf.empty:
                continue
            errs = (cdf["parsed_value"] - cdf["true_value"]).abs() / cdf["true_value"].abs() * 100
            ai_sub = ai_df[(ai_df["model"] == model) & (ai_df["condition"] == cond)]["anchoring_index"].dropna()
            ai_str = f"{ai_sub.mean():.2f}±{ai_sub.sem():.2f}" if len(ai_sub) > 0 else "-"
            print(f"{CONDITION_LABELS[cond]:18s} {len(cdf):6d} {errs.median():7.1f}% {errs.mean():8.1f}% {ai_str:>12s}")

    print("\n" + "=" * 70)
    print("TABLE 2: Wilcoxon Signed-Rank Tests (per-question medians)")
    print("=" * 70)
    for model in sorted(df["model"].unique()):
        mdf = df[df["model"] == model]
        print(f"\n{'Comparison':30s} {'W':>8s} {'p':>10s} {'Med.Diff':>10s} {'Sig':>6s}")
        print("-" * 66)
        for cond in ["high_anchor", "low_anchor", "counter_anchor"]:
            b_meds, c_meds = [], []
            for qid in mdf["question_id"].unique():
                qdf = mdf[mdf["question_id"] == qid]
                b = qdf[qdf["condition"] == "baseline"]["parsed_value"].median()
                c = qdf[qdf["condition"] == cond]["parsed_value"].median()
                if pd.notna(b) and pd.notna(c):
                    b_meds.append(b)
                    c_meds.append(c)
            diffs = np.array(c_meds) - np.array(b_meds)
            nz = diffs[diffs != 0]
            if len(nz) >= 5:
                w, p = stats.wilcoxon(nz)
                sig = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "(+)" if p < .1 else "n.s."
                print(f"{'Baseline vs ' + CONDITION_LABELS[cond]:30s} {w:8.1f} {p:10.6f} {np.median(diffs):+9.2f} {sig:>6s}")
            else:
                print(f"{'Baseline vs ' + CONDITION_LABELS[cond]:30s} {'—':>8s} {'—':>10s} {'—':>10s} {'—':>6s}")


# ── Main ────────────────────────────────────────────────────────────────
def main():
    print("Loading data...")
    df = load_clean_data()
    print(f"  {len(df)} records, {df['question_id'].nunique()} questions, "
          f"{df['model'].nunique()} model(s)")

    ai_df = compute_ai(df)

    print("\nGenerating paper figures...")
    fig1_boxplots(df)
    fig2_anchoring_index(ai_df)
    fig3_heatmap(ai_df)
    fig4_counter_effectiveness(df)
    fig5_scatter(df)
    fig6_direction_summary(df)

    print_summary_table(df, ai_df)

    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
