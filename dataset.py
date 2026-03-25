"""
PTF (Play the Future) dataset for anchoring bias experiments.

Source: JiaxuLou/LLM_Bias (https://github.com/JiaxuLou/LLM_Bias)
Paper: Lou & Sun (2024), "Anchoring Bias in Large Language Models: An Experimental Study"

49 prediction questions with verified ground truth, each with:
- hint_1: factual reference info
- high_anchor / low_anchor: PTF-native anchoring hints
- true_value: actual outcome

Run convert_ptf.py first to generate dataset_ptf.json from the raw data.
"""

import json
from pathlib import Path

_PTF_PATH = Path(__file__).parent / "dataset_ptf.json"


def load_dataset() -> list[dict]:
    """Load the PTF dataset from JSON."""
    if not _PTF_PATH.exists():
        raise FileNotFoundError(
            f"{_PTF_PATH} not found. Run 'python convert_ptf.py' first."
        )
    with open(_PTF_PATH, encoding="utf-8") as f:
        return json.load(f)


# For backward compatibility
DATASET = load_dataset() if _PTF_PATH.exists() else []

if __name__ == "__main__":
    data = load_dataset()
    print(f"Total questions: {len(data)}")

    from collections import Counter
    cats = Counter(q["category"] for q in data)
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    print(f"\nSample question:")
    q = data[0]
    print(f"  ID: {q['id']}")
    print(f"  Q: {q['question'][:120]}...")
    print(f"  True value: {q['true_value']} {q['unit']}")
    print(f"  Low anchor: {q['low_anchor']} ({q['low_anchor_text'][:60]}...)")
    print(f"  High anchor: {q['high_anchor']} ({q['high_anchor_text'][:60]}...)")
