from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def nested_get(row: Dict[str, Any], key: str) -> Optional[float]:
    value: Any = row
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    if isinstance(value, (int, float)):
        return float(value)
    return None


def best_row(rows: List[Dict[str, Any]], metric: str) -> Optional[Dict[str, Any]]:
    candidates = [(nested_get(row, metric), row) for row in rows]
    candidates = [(score, row) for score, row in candidates if score is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def compact_row(row: Dict[str, Any]) -> Dict[str, Any]:
    val = row.get("val", {})
    return {
        "epoch": row.get("epoch"),
        "val": {
            "accuracy": val.get("accuracy"),
            "balanced_accuracy": val.get("balanced_accuracy"),
            "macro_f1": val.get("macro_f1"),
            "top2_accuracy": val.get("top2_accuracy"),
            "top3_accuracy": val.get("top3_accuracy"),
            "oracle_gain": val.get("oracle_gain"),
        },
    }


def summarize(metrics_path: Path) -> Dict[str, Any]:
    rows = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary: Dict[str, Any] = {
        "metrics_path": str(metrics_path),
        "epochs": len(rows),
    }
    for metric in [
        "val.accuracy",
        "val.balanced_accuracy",
        "val.macro_f1",
        "val.top2_accuracy",
        "val.top3_accuracy",
        "val.oracle_gain.mean_predicted_score",
        "val.oracle_gain.positive_gain_rate",
        "val.oracle_gain.mean_predicted_relative_gain",
        "val.oracle_gain.mean_gain_ratio_when_best_positive",
    ]:
        row = best_row(rows, metric)
        if row is not None:
            summary[f"best_{metric.replace('.', '_')}"] = compact_row(row)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.metrics), indent=2))


if __name__ == "__main__":
    main()
