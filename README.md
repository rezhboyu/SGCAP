# SGCAP: Self-Generated Counter-Anchor Prompting

Mitigating numerical anchoring bias in Large Language Models through self-generated counter-anchors.

## Overview

LLMs exhibit **anchoring bias** — a systematic tendency to over-rely on initial numerical values when producing estimates. This project proposes **Self-Generated Counter-Anchor Prompting (SGCAP)**, a three-stage framework where the model autonomously generates extreme opposing estimates to serve as a balanced reference frame before producing a debiased final estimate.

### Key Findings

- GPT-4o-mini exhibits significant anchoring bias across 44 prediction tasks (mean Anchoring Index: 0.41 high / 0.30 low)
- 61% of questions show at least one direction of anchoring
- SGCAP with dual-anchor awareness reduces median absolute percentage error from 9.7% (baseline) to 8.3%

## Dataset

We use the **Play-the-Future (PTF) dataset** (Yasseri & Reher, 2022), adapted from [Lou & Sun (2025)](https://github.com/JiaxuLou/LLM_Bias). 44 prediction questions across 7 domains: Stock & Finance, Temperature, Social Media, Counting, Sports Statistics, Flights, and Film Revenue.

## Experimental Conditions

| Condition | Description |
|-----------|-------------|
| Baseline | Factual reference hint only, no anchoring |
| High Anchor | Reference hint + high anchor |
| Low Anchor | Reference hint + low anchor |
| Counter-Anchor | Both anchors + explicit debiasing instruction |
| SGCAP v1 | Self-generated counter-anchor prompting |
| SGCAP v2 | SGCAP with dual-anchor awareness |

## Project Structure

```
├── experiment.py        # Experiment runner (API calls to LLMs)
├── analysis.py          # Statistical analysis & visualization
├── paper_figures.py     # Publication-quality figure generation
├── compare_figures.py   # Comparison visualizations
├── dataset.py           # PTF dataset loader
├── convert_ptf.py       # Raw PTF data → structured JSON
├── dataset_ptf.json     # Processed dataset (44 questions)
├── raw_ptf_data.json    # Original PTF data
├── results/             # Experiment outputs (JSONL)
├── figures/             # Generated plots (PNG)
├── paper_draft.md       # Paper manuscript
└── requirements.txt     # Python dependencies
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up OpenAI API key
echo "OPENAI_API_KEY=sk-..." > .env

# Run experiment (dry run first)
python experiment.py --dry-run
python experiment.py --model gpt-4o-mini --trials 5  # quick test

# Full experiment
python experiment.py

# Generate analysis & figures
python analysis.py
python paper_figures.py
```

## Usage

```bash
# Single model / condition
python experiment.py --model gpt-4o-mini --condition baseline

# Limit questions for testing
python experiment.py --questions 3 --trials 5

# Analyze specific model
python analysis.py --model gpt-4o-mini
```

## Sample Figures

| Boxplots by Condition | Anchoring Index |
|:---:|:---:|
| ![Boxplots](figures/fig1_boxplots.png) | ![Anchoring Index](figures/fig2_anchoring_index.png) |

| Heatmap | Counter-Anchor Effectiveness |
|:---:|:---:|
| ![Heatmap](figures/fig3_heatmap_gpt-4o-mini.png) | ![Counter](figures/fig4_counter_effectiveness.png) |

## References

- Tversky, A., & Kahneman, D. (1974). Judgment under Uncertainty: Heuristics and Biases. *Science*, 185(4157), 1124–1131.
- Lou, J., & Sun, Z. (2025). Anchoring Bias in Large Language Models: An Experimental Study.
- Yasseri, T., & Reher, J. (2022). Play-the-Future Dataset.

## License

This project is for academic research purposes.
