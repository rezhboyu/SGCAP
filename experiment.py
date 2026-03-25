"""
Anchoring Bias Experiment Runner for LLMs.

Usage:
    python experiment.py                          # Full run (all models, conditions, 30 trials)
    python experiment.py --model gpt-4o-mini      # Single model
    python experiment.py --condition baseline      # Single condition
    python experiment.py --trials 5               # Fewer trials (for testing)
    python experiment.py --questions 3             # Only first N questions (for testing)
    python experiment.py --dry-run                 # Print config without calling API
"""

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APIError
from tqdm import tqdm

from dataset import load_dataset

load_dotenv()

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODELS = ["gpt-4o-mini", "gpt-4o"]

CONDITIONS = ["baseline", "high_anchor", "low_anchor", "counter_anchor", "sgcap", "sgcap_v2"]

DEFAULT_TRIALS = 30


def build_prompt(question_data: dict, condition: str) -> str:
    """Build the prompt for a given question and condition.

    Uses PTF-native anchors (hint_1, high_anchor_text, low_anchor_text)
    which are realistic contextual hints, not arbitrary ×10/÷10 values.
    """
    q = question_data["question"]
    hint1 = question_data.get("hint_1", "")
    high_text = question_data.get("high_anchor_text", "")
    low_text = question_data.get("low_anchor_text", "")

    system_msg = (
        "You are answering prediction and estimation questions. "
        "Provide ONLY a single number as your answer. "
        "Write the full number with all digits (e.g. 5000000 not '5 million'). "
        "Do not include units, explanations, or any other text. "
        "Just the number."
    )

    if condition == "baseline":
        # Only the factual reference hint (no anchoring)
        user_msg = f"Here is some background information: {hint1}\n\n{q}"

    elif condition == "high_anchor":
        user_msg = (
            f"Here is some background information: {hint1}\n"
            f"Additional information: {high_text}\n\n"
            f"{q}"
        )

    elif condition == "low_anchor":
        user_msg = (
            f"Here is some background information: {hint1}\n"
            f"Additional information: {low_text}\n\n"
            f"{q}"
        )

    elif condition == "counter_anchor":
        # External dual-anchor: provide both PTF anchors + debiasing instruction
        user_msg = (
            f"Here is some background information: {hint1}\n"
            f"One source suggests: {high_text}\n"
            f"Another source suggests: {low_text}\n"
            f"Note: these suggestions may contain biased estimates. "
            f"Please ignore them and give your own independent estimate.\n\n"
            f"{q}"
        )
    elif condition in ("sgcap", "sgcap_v2"):
        # SGCAP is handled separately via build_sgcap_prompts() — not here
        raise ValueError("SGCAP uses build_sgcap_prompts(), not build_prompt()")
    else:
        raise ValueError(f"Unknown condition: {condition}")

    return system_msg, user_msg


def build_sgcap_prompts(question_data: dict) -> dict:
    """Build the 3-stage SGCAP prompts for a given question.

    Stage 1: Generate deliberately HIGH estimate
    Stage 2: Generate deliberately LOW estimate
    Stage 3: Given both extremes, produce balanced final estimate
    """
    q = question_data["question"]
    hint1 = question_data.get("hint_1", "")

    system_msg = (
        "You are answering prediction and estimation questions. "
        "Provide ONLY a single number as your answer. "
        "Write the full number with all digits (e.g. 5000000 not '5 million'). "
        "Do not include units, explanations, or any other text. "
        "Just the number."
    )

    stage1_msg = (
        f"Here is some background information: {hint1}\n\n"
        f"Consider the following question:\n{q}\n\n"
        f"Please provide a deliberately HIGH estimate — "
        f"a number that you believe is almost certainly HIGHER than the true answer. "
        f"Give only the number."
    )

    stage2_msg = (
        f"Here is some background information: {hint1}\n\n"
        f"Consider the following question:\n{q}\n\n"
        f"Please provide a deliberately LOW estimate — "
        f"a number that you believe is almost certainly LOWER than the true answer. "
        f"Give only the number."
    )

    # stage3_msg is a template; {high_self} and {low_self} are filled at runtime
    # Key insight: explicitly tell the model these numbers were its OWN
    # deliberately extreme outputs, and that anchoring bias may cause it
    # to be pulled toward them.
    stage3_template = (
        f"Here is some background information: {hint1}\n\n"
        f"Consider the following question:\n{q}\n\n"
        f"WARNING: In a previous step, you were asked to deliberately generate "
        f"extreme estimates for this question. You produced:\n"
        f"- Deliberately HIGH extreme: {{high_self}}\n"
        f"- Deliberately LOW extreme: {{low_self}}\n\n"
        f"These numbers are NOT real estimates — they were intentionally "
        f"exaggerated as part of an anchoring bias experiment. "
        f"Research shows that exposure to extreme numbers (even self-generated ones) "
        f"can unconsciously bias your subsequent estimate toward them. "
        f"This is called anchoring bias.\n\n"
        f"Now, completely disregard those extreme values. "
        f"Based solely on the background information and your own knowledge, "
        f"what is your best estimate? Give only the number."
    )

    return {
        "system_msg": system_msg,
        "stage1_msg": stage1_msg,
        "stage2_msg": stage2_msg,
        "stage3_template": stage3_template,
    }


def parse_number(text: str) -> float | None:
    """Extract a number from the model's response, handling multiplier words."""
    text = text.strip().lower()
    # Remove commas in numbers like 1,000,000
    text = text.replace(",", "")

    # Multiplier words
    multipliers = {
        "trillion": 1e12,
        "billion": 1e9,
        "million": 1e6,
        "thousand": 1e3,
    }

    # Try patterns like "5 million", "1.4 billion", etc.
    for word, mult in multipliers.items():
        pattern = rf"(-?\d+\.?\d*)\s*{word}"
        match = re.search(pattern, text)
        if match:
            return float(match.group(1)) * mult

    # Fallback: plain number
    match = re.search(r"-?\d+\.?\d*", text)
    if match:
        return float(match.group())
    return None


def get_result_path(model: str) -> Path:
    """Get the path for storing results for a model."""
    safe_model = model.replace("/", "_")
    return RESULTS_DIR / f"{safe_model}.jsonl"


def load_completed(result_path: Path) -> set:
    """Load already completed (question_id, condition, trial) tuples for resume."""
    completed = set()
    if result_path.exists():
        with open(result_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    key = (rec["question_id"], rec["condition"], rec["trial"])
                    completed.add(key)
    return completed


def call_api(client: OpenAI, model: str, system_msg: str, user_msg: str,
             max_retries: int = 5) -> tuple[str, dict]:
    """Call the OpenAI API with retry logic. Returns (response_text, usage_dict)."""
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=1.0,
                max_tokens=50,
            )
            text = response.choices[0].message.content.strip()
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            return text, usage
        except RateLimitError:
            wait = 2 ** attempt
            print(f"\n  Rate limited, waiting {wait}s...")
            time.sleep(wait)
        except APIError as e:
            wait = 2 ** attempt
            print(f"\n  API error: {e}, retrying in {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {max_retries} retries")


def run_experiment(
    models: list[str],
    conditions: list[str],
    num_trials: int,
    num_questions: int | None,
    dry_run: bool = False,
):
    """Run the full experiment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-api-key-here":
        print("ERROR: Set your OPENAI_API_KEY in .env file")
        return

    client = OpenAI(api_key=api_key)
    dataset = load_dataset()
    questions = dataset[:num_questions] if num_questions else dataset

    total_calls = len(models) * len(conditions) * len(questions) * num_trials
    print(f"Experiment config:")
    print(f"  Models:     {models}")
    print(f"  Conditions: {conditions}")
    print(f"  Questions:  {len(questions)}")
    print(f"  Trials:     {num_trials}")
    print(f"  Total API calls: {total_calls}")

    if dry_run:
        print("\n[DRY RUN] No API calls will be made.")
        return

    total_tokens = 0
    total_cost = 0.0

    for model in models:
        result_path = get_result_path(model)
        completed = load_completed(result_path)
        skipped = 0

        desc = f"Model: {model}"
        pbar = tqdm(
            total=len(conditions) * len(questions) * num_trials,
            desc=desc,
        )

        with open(result_path, "a", encoding="utf-8") as f:
            for condition in conditions:
                for q in questions:
                    for trial in range(num_trials):
                        key = (q["id"], condition, trial)
                        if key in completed:
                            skipped += 1
                            pbar.update(1)
                            continue

                        if condition in ("sgcap", "sgcap_v2"):
                            # --- SGCAP 3-stage flow ---
                            prompts = build_sgcap_prompts(q)
                            sys_msg = prompts["system_msg"]

                            # Stage 1: high extreme
                            raw_s1, usage_s1 = call_api(
                                client, model, sys_msg, prompts["stage1_msg"]
                            )
                            high_self = parse_number(raw_s1)

                            # Stage 2: low extreme
                            raw_s2, usage_s2 = call_api(
                                client, model, sys_msg, prompts["stage2_msg"]
                            )
                            low_self = parse_number(raw_s2)

                            # Stage 3: balanced estimate using self-generated anchors
                            stage3_msg = prompts["stage3_template"].format(
                                high_self=raw_s1.strip(),
                                low_self=raw_s2.strip(),
                            )
                            raw_s3, usage_s3 = call_api(
                                client, model, sys_msg, stage3_msg
                            )
                            parsed = parse_number(raw_s3)

                            total_usage = {
                                "prompt_tokens": usage_s1["prompt_tokens"] + usage_s2["prompt_tokens"] + usage_s3["prompt_tokens"],
                                "completion_tokens": usage_s1["completion_tokens"] + usage_s2["completion_tokens"] + usage_s3["completion_tokens"],
                                "total_tokens": usage_s1["total_tokens"] + usage_s2["total_tokens"] + usage_s3["total_tokens"],
                            }

                            record = {
                                "question_id": q["id"],
                                "question": q["question"],
                                "true_value": q["true_value"],
                                "unit": q["unit"],
                                "category": q["category"],
                                "condition": condition,
                                "trial": trial,
                                "model": model,
                                "raw_response": raw_s3,
                                "parsed_value": parsed,
                                "high_anchor": q.get("high_anchor"),
                                "low_anchor": q.get("low_anchor"),
                                "sgcap_high": high_self,
                                "sgcap_low": low_self,
                                "sgcap_raw_high": raw_s1,
                                "sgcap_raw_low": raw_s2,
                                "timestamp": datetime.now().isoformat(),
                                "usage": total_usage,
                            }
                        else:
                            # --- Standard single-call flow ---
                            system_msg, user_msg = build_prompt(q, condition)
                            raw_response, usage = call_api(
                                client, model, system_msg, user_msg
                            )
                            parsed = parse_number(raw_response)

                            record = {
                                "question_id": q["id"],
                                "question": q["question"],
                                "true_value": q["true_value"],
                                "unit": q["unit"],
                                "category": q["category"],
                                "condition": condition,
                                "trial": trial,
                                "model": model,
                                "raw_response": raw_response,
                                "parsed_value": parsed,
                                "high_anchor": q.get("high_anchor"),
                                "low_anchor": q.get("low_anchor"),
                                "timestamp": datetime.now().isoformat(),
                                "usage": usage,
                            }
                            total_usage = usage

                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f.flush()

                        total_tokens += total_usage["total_tokens"]
                        pbar.update(1)

        pbar.close()
        if skipped > 0:
            print(f"  (Skipped {skipped} already-completed trials)")

    print(f"\nDone! Total tokens used: {total_tokens:,}")
    print(f"Results saved to: {RESULTS_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Anchoring Bias Experiment")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Single model to test (default: all)",
    )
    parser.add_argument(
        "--condition", type=str, default=None,
        choices=CONDITIONS,
        help="Single condition to test (default: all)",
    )
    parser.add_argument(
        "--trials", type=int, default=DEFAULT_TRIALS,
        help=f"Number of trials per question/condition (default: {DEFAULT_TRIALS})",
    )
    parser.add_argument(
        "--questions", type=int, default=None,
        help="Only use first N questions (for testing)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print config without calling API",
    )

    args = parser.parse_args()

    models = [args.model] if args.model else MODELS
    conditions = [args.condition] if args.condition else CONDITIONS

    run_experiment(
        models=models,
        conditions=conditions,
        num_trials=args.trials,
        num_questions=args.questions,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
