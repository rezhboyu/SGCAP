"""
Convert raw PTF (Play the Future) JSON from JiaxuLou/LLM_Bias
into our dataset.py format.

Handles various result formats (USD, °F, °C, views, h:mm, etc.)
and extracts hint_2_a (anchor_a) and hint_2_b (anchor_b) for each question.
"""

import json
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")


def parse_ptf_number(text: str) -> float | None:
    """Parse a PTF result/hint value into a float."""
    text = str(text).replace("\xa0", " ").replace(",", "").strip()

    # Handle time formats like "2:15 h", "06:14 h", "3:37 mins", "11:29 mins"
    time_match = re.match(r"(\d+):(\d+)\s*(h|hrs?|mins?|secs?)", text)
    if time_match:
        hours_or_mins = float(time_match.group(1))
        sub = float(time_match.group(2))
        unit = time_match.group(3)
        if unit.startswith("h"):
            return hours_or_mins * 60 + sub  # convert to minutes
        elif unit.startswith("min"):
            return hours_or_mins * 60 + sub  # already mm:ss -> seconds
        elif unit.startswith("sec"):
            return hours_or_mins + sub / 100  # seconds.centiseconds

    # Handle "226.89 secs"
    secs_match = re.match(r"([\d.]+)\s*secs?", text)
    if secs_match:
        return float(secs_match.group(1))

    # Handle "157.3 Mio. USD" (millions)
    mio_match = re.search(r"([\d.]+)\s*Mio", text)
    if mio_match:
        return float(mio_match.group(1)) * 1_000_000

    # General: extract first number
    text_clean = text.replace("$", "").replace("€", "")
    match = re.search(r"-?[\d]+\.?\d*", text_clean)
    if match:
        return float(match.group())

    return None


def extract_anchor_value(hint_text: str) -> float | None:
    """Extract the numeric anchor value from a hint string.

    PTF hints follow the pattern 'Description: VALUE UNIT'.
    We extract the value AFTER the last colon, which is the actual anchor.
    """
    text = str(hint_text).replace("\xa0", " ").replace(",", "").strip()

    # Split on the last colon to get the value part
    if ":" in text:
        # Find the last segment that looks like "VALUE UNIT"
        parts = text.rsplit(":", 1)
        value_part = parts[-1].strip()
        # But check if it's a time format like "2:19" after the colon
        # In that case the colon is part of the time, not a separator
        # Detect: if value_part starts with digits and the char before ':' is a digit
        if parts[0] and parts[0][-1].isdigit() and value_part and value_part[0].isdigit():
            # Could be time format "2:19 h" — check if there are earlier colons
            earlier_parts = parts[0].rsplit(":", 1)
            if len(earlier_parts) > 1:
                # Use the part after the second-to-last colon
                time_str = earlier_parts[-1].strip() + ":" + value_part
                return parse_ptf_number(time_str)
            else:
                # The whole thing might be "something: 2:19 h"
                return parse_ptf_number(value_part)
        else:
            return parse_ptf_number(value_part)

    return parse_ptf_number(text)


def main():
    with open("raw_ptf_data.json", encoding="utf-8") as f:
        data = json.load(f)

    questions = {}
    for item in data:
        inner = json.loads(item["input"])
        qid = inner["question_id"]
        if qid not in questions:
            questions[qid] = inner

    print(f"Total unique questions: {len(questions)}\n")

    dataset = []
    skipped = []

    for qid in sorted(questions.keys()):
        q = questions[qid]
        result_str = str(q.get("Result", "")).replace("\xa0", " ")
        true_val = parse_ptf_number(result_str)

        hint1_str = str(q.get("hint_1", "")).replace("\xa0", " ")
        hint2a_str = str(q.get("hint_2_a", "")).replace("\xa0", " ")
        hint2b_str = str(q.get("hint_2_b", "")).replace("\xa0", " ")

        anchor_a = extract_anchor_value(hint2a_str)
        anchor_b = extract_anchor_value(hint2b_str)
        reference = extract_anchor_value(hint1_str)

        # Determine unit from result string
        unit_match = re.search(
            r"(USD|EUR|RUB|CHF|CAD|°[FC]|views|likes|people|words|"
            r"times|songs|racers|subscribers|reactions|upvotes|"
            r"passes|flights|secs|mins|h)\b",
            result_str,
        )
        unit = unit_match.group(1) if unit_match else ""

        # Skip if true_value is 0 or None (can't do meaningful anchoring)
        if true_val is None or true_val == 0:
            skipped.append((qid, q["event_name"], result_str, "true_val=0 or unparseable"))
            continue

        # Skip time-format questions (hard to anchor numerically)
        if re.search(r"\d+:\d+\s*(h|hrs?|mins?|secs?)", result_str):
            skipped.append((qid, q["event_name"], result_str, "time format"))
            continue
        # Also skip if result contains "secs" (lap times etc)
        if "secs" in result_str.lower():
            skipped.append((qid, q["event_name"], result_str, "seconds format"))
            continue

        # Skip if we can't parse anchors
        if anchor_a is None or anchor_b is None:
            skipped.append((qid, q["event_name"], result_str, "anchor unparseable"))
            continue

        user_question = str(q["user_question"]).replace("\xa0", " ").replace("\u2019", "'")

        entry = {
            "id": f"ptf_{qid:02d}",
            "question": user_question,
            "true_value": true_val,
            "unit": unit,
            "category": q["event_name"],
            "hint_1": hint1_str,
            "anchor_a": anchor_a,
            "anchor_a_text": hint2a_str,
            "anchor_b": anchor_b,
            "anchor_b_text": hint2b_str,
            "reference_value": reference,
        }
        dataset.append(entry)

    print(f"Converted: {len(dataset)} questions")
    print(f"Skipped: {len(skipped)} questions")
    for qid, name, result, reason in skipped:
        print(f"  Q{qid}: [{name}] result='{result}' -> {reason}")

    # Determine which anchor is high/low for each question
    for entry in dataset:
        if entry["anchor_a"] > entry["anchor_b"]:
            entry["high_anchor"] = entry["anchor_a"]
            entry["high_anchor_text"] = entry["anchor_a_text"]
            entry["low_anchor"] = entry["anchor_b"]
            entry["low_anchor_text"] = entry["anchor_b_text"]
        else:
            entry["high_anchor"] = entry["anchor_b"]
            entry["high_anchor_text"] = entry["anchor_b_text"]
            entry["low_anchor"] = entry["anchor_a"]
            entry["low_anchor_text"] = entry["anchor_a_text"]

    # Write to dataset_ptf.json
    with open("dataset_ptf.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to dataset_ptf.json ({len(dataset)} questions)")

    # Print sample
    print("\n=== Sample entries ===")
    for entry in dataset[:5]:
        print(f"{entry['id']}: true={entry['true_value']}, "
              f"low={entry['low_anchor']}, high={entry['high_anchor']}, "
              f"unit={entry['unit']}")
        print(f"  Q: {entry['question'][:100]}...")
        print()


if __name__ == "__main__":
    main()
