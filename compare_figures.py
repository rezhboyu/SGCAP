"""Generate comparison figures: External anchors vs Self-generated vs High/Low."""

import json
import sys

sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "SimHei", "Arial"]
matplotlib.rcParams["axes.unicode_minus"] = False

from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"

# Load & clean
records = []
with open(RESULTS_DIR / "gpt-4o-mini.jsonl", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))
df = pd.DataFrame(records).dropna(subset=["parsed_value"])

EXCLUDE = {"ptf_11", "ptf_18", "ptf_45", "ptf_54", "ptf_55"}
df = df[~df["question_id"].isin(EXCLUDE)]

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


# ============================================
# Figure 1: 3-panel scatter comparison
# ============================================
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

comparisons = [
    ("high_anchor", "counter_anchor", "High Anchor vs Counter-Anchor\n(External Dual-Anchor)"),
    ("high_anchor", "sgcap_v2", "High Anchor vs SGCAP v2\n(Self-Generated Anchor)"),
    ("counter_anchor", "sgcap_v2", "Counter-Anchor (External)\nvs SGCAP v2 (Self-Generated)"),
]

LABEL_MAP = {
    "high_anchor": "High Anchor",
    "low_anchor": "Low Anchor",
    "counter_anchor": "Counter-Anchor",
    "sgcap_v2": "SGCAP v2",
}

for ax, (cond_x, cond_y, title) in zip(axes, comparisons):
    x_errors, y_errors = [], []
    for qid in df["question_id"].unique():
        qdf = df[df["question_id"] == qid]
        tv = qdf["true_value"].iloc[0]
        xm = qdf[qdf["condition"] == cond_x]["parsed_value"].median()
        ym = qdf[qdf["condition"] == cond_y]["parsed_value"].median()
        if not np.isnan(xm) and not np.isnan(ym) and tv != 0:
            x_errors.append(abs(xm - tv) / abs(tv) * 100)
            y_errors.append(abs(ym - tv) / abs(tv) * 100)

    ax.scatter(x_errors, y_errors, alpha=0.6, s=40, c="#333", edgecolors="white", linewidth=0.5)
    max_val = max(max(x_errors, default=1), max(y_errors, default=1)) * 1.1
    clip = min(max_val, 300)
    ax.plot([0, clip], [0, clip], "k--", linewidth=0.8, alpha=0.5)

    above = sum(1 for x, y in zip(x_errors, y_errors) if y > x)
    below = sum(1 for x, y in zip(x_errors, y_errors) if y < x)

    ax.fill_between([0, clip], [0, clip], [clip, clip], alpha=0.06, color="red")
    ax.fill_between([0, clip], [0, 0], [0, clip], alpha=0.06, color="green")

    ax.text(0.95, 0.05, f"Y wins: {below}", transform=ax.transAxes, ha="right", fontsize=11, color="green", fontweight="bold")
    ax.text(0.05, 0.95, f"X wins: {above}", transform=ax.transAxes, ha="left", va="top", fontsize=11, color="red", fontweight="bold")

    ax.set_xlabel(f"{LABEL_MAP[cond_x]} |% Error|", fontsize=11)
    ax.set_ylabel(f"{LABEL_MAP[cond_y]} |% Error|", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0, clip)
    ax.set_ylim(0, clip)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "comparison_scatter_3panel.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIGURES_DIR / 'comparison_scatter_3panel.png'}")


# ============================================
# Figure 2: Domain breakdown grouped bar
# ============================================
def categorize(row):
    unit = str(row["unit"])
    if unit in ("USD", "EUR", "RUB", "CHF", "CAD"):
        return "Stock/Finance"
    elif "F" in unit or "C" in unit:
        return "Temperature"
    elif unit in ("views", "likes", "subscribers", "reactions", "upvotes"):
        return "Social Media"
    elif unit in ("flights",):
        return "Flights"
    elif unit in ("passes",):
        return "Sports"
    elif unit in ("words", "times", "songs", "people", "racers"):
        return "Counting"
    else:
        return "Other"


df["domain"] = df.apply(categorize, axis=1)

conds = ["baseline", "high_anchor", "low_anchor", "counter_anchor", "sgcap_v2"]
labels = ["Baseline", "High Anchor", "Low Anchor", "Counter\n(External)", "SGCAP v2\n(Self)"]
colors = ["#4C72B0", "#DD5555", "#55AA55", "#D4A017", "#E91E90"]

domains = sorted(df["domain"].unique())
fig, ax = plt.subplots(figsize=(14, 6))

x = np.arange(len(domains))
width = 0.15

for i, (cond, label, color) in enumerate(zip(conds, labels, colors)):
    offset = (i - len(conds) / 2 + 0.5) * width
    mdapes = []
    for domain in domains:
        ddf = df[(df["domain"] == domain) & (df["condition"] == cond)]
        if ddf.empty:
            mdapes.append(0)
        else:
            errors = abs(ddf["parsed_value"] - ddf["true_value"]) / abs(ddf["true_value"]) * 100
            mdapes.append(errors.median())
    ax.bar(x + offset, mdapes, width, label=label, color=color, edgecolor="white", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(domains, fontsize=10)
ax.set_ylabel("MdAPE (%)")
ax.set_title("Estimation Error by Task Domain and Debiasing Method")
ax.legend(fontsize=9, ncol=5, loc="upper left")
ax.set_ylim(0)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "domain_comparison_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {FIGURES_DIR / 'domain_comparison_bar.png'}")


# ============================================
# Print summary
# ============================================
print("\n" + "=" * 80)
print("COMPARISON: External Dual-Anchor vs Self-Generated (SGCAP v2)")
print("=" * 80)
for cond, label in [
    ("baseline", "Baseline"),
    ("high_anchor", "High Anchor"),
    ("low_anchor", "Low Anchor"),
    ("counter_anchor", "Counter-Anchor (External)"),
    ("sgcap", "SGCAP v1 (naive)"),
    ("sgcap_v2", "SGCAP v2 (bias-aware)"),
]:
    cdf = df[df["condition"] == cond]
    if cdf.empty:
        continue
    errors = abs(cdf["parsed_value"] - cdf["true_value"]) / abs(cdf["true_value"]) * 100
    print(
        f"  {label:30s}: n={len(cdf):5d}, "
        f"MdAPE={errors.median():6.1f}%, "
        f"Mean={errors.mean():7.1f}%, "
        f"Std={errors.std():7.1f}%"
    )
