from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def nested_get(row: Dict[str, Any], key: str, default: Any = None) -> Any:
    value: Any = row
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def summarize_one(path: Path) -> Dict[str, Any]:
    metrics = json.loads(path.read_text(encoding="utf-8"))
    stay = nested_get(metrics, "per_action.Stay", {})
    return {
        "run": path.stem,
        "path": str(path),
        "accuracy": nested_get(metrics, "accuracy"),
        "balanced_accuracy": nested_get(metrics, "balanced_accuracy"),
        "macro_f1": nested_get(metrics, "macro_f1"),
        "top2_accuracy": nested_get(metrics, "top2_accuracy"),
        "top3_accuracy": nested_get(metrics, "top3_accuracy"),
        "mean_predicted_score": nested_get(metrics, "oracle_gain.mean_predicted_score"),
        "positive_gain_rate": nested_get(metrics, "oracle_gain.positive_gain_rate"),
        "nonnegative_gain_rate": nested_get(metrics, "oracle_gain.nonnegative_gain_rate"),
        "negative_gain_rate": nested_get(metrics, "oracle_gain.negative_gain_rate"),
        "safe_balanced_gain_score": nested_get(metrics, "safe_balanced_gain_score"),
        "stay_precision": stay.get("precision"),
        "stay_recall": stay.get("recall"),
        "stay_predicted_count": stay.get("predicted_count"),
        "selection": metrics.get("selection", {}),
    }


def markdown_table(rows: Iterable[Dict[str, Any]]) -> str:
    columns = [
        ("Run", "run"),
        ("Acc", "accuracy"),
        ("Macro F1", "macro_f1"),
        ("Mean Gain", "mean_predicted_score"),
        ("Pos Gain", "positive_gain_rate"),
        ("Neg Gain", "negative_gain_rate"),
        ("Safe Score", "safe_balanced_gain_score"),
        ("Stay Pred", "stay_predicted_count"),
        ("Stay Recall", "stay_recall"),
    ]
    lines = [
        "| " + " | ".join(name for name, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for _, key in columns:
            value = row.get(key)
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            elif value is None:
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    rows = [summarize_one(path) for path in args.files]
    rows = sorted(rows, key=lambda row: row["run"])
    text = json.dumps(rows, indent=2)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown_table(rows), encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
