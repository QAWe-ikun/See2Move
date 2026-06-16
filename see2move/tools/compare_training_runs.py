from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from see2move.tools.summarize_training_run import best_row, nested_get


DEFAULT_PATTERNS = [
    "runs/ai2thor_policy_10k_stay_simple_e35/metrics.json",
    "runs/ai2thor_policy_10k_stay_qwen3vl_e20/metrics.json",
    "runs/ai2thor_policy_10k_stay_qwen3vl_action_e25/metrics.json",
    "runs/ai2thor_policy_10k_stay_qwen3vl_action_mild_e25/metrics.json",
    "runs/ai2thor_policy_10k_stay_qwen3vl_gain_score_e25/metrics.json",
    "runs/ai2thor_policy_10k_stay_qwen3vl_gated_e20/metrics.json",
    "runs/ai2thor_policy_4k8_simple_e35/metrics.json",
    "runs/ai2thor_policy_4k8_qwen3vl_depth_e20/metrics.json",
    "runs/ai2thor_policy_4k8_qwen3vl_depth_gain_e20/metrics.json",
    "runs/ablations_10k_stay/*/metrics.json",
    "runs/ablations_4k8/*/metrics.json",
]


def expand_inputs(inputs: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    seen = set()
    for item in inputs:
        matches = glob.glob(item)
        if not matches:
            matches = [item]
        for match in matches:
            path = Path(match)
            if path.is_dir():
                path = path / "metrics.json"
            if not path.exists() or path in seen:
                continue
            seen.add(path)
            paths.append(path)
    return sorted(paths, key=lambda path: str(path))


def score(rows: List[Dict[str, Any]], metric: str) -> float | None:
    row = best_row(rows, metric)
    if row is None:
        return None
    return nested_get(row, metric)


def summarize_one(metrics_path: Path) -> Dict[str, Any]:
    rows = json.loads(metrics_path.read_text(encoding="utf-8"))
    result = {
        "run": str(metrics_path.parent),
        "metrics_path": str(metrics_path),
        "epochs": len(rows),
    }
    metrics = {
        "best_accuracy": "val.accuracy",
        "best_balanced_accuracy": "val.balanced_accuracy",
        "best_macro_f1": "val.macro_f1",
        "best_balanced_gain_score": "val.balanced_gain_score",
        "best_safe_balanced_gain_score": "val.safe_balanced_gain_score",
        "best_top2_accuracy": "val.top2_accuracy",
        "best_top3_accuracy": "val.top3_accuracy",
        "best_mean_predicted_score": "val.oracle_gain.mean_predicted_score",
        "best_positive_gain_rate": "val.oracle_gain.positive_gain_rate",
        "best_nonnegative_gain_rate": "val.oracle_gain.nonnegative_gain_rate",
        "best_negative_gain_rate": "val.oracle_gain.negative_gain_rate",
        "best_mean_predicted_relative_gain": "val.oracle_gain.mean_predicted_relative_gain",
    }
    for name, metric in metrics.items():
        result[name] = score(rows, metric)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", nargs="*")
    args = parser.parse_args()
    inputs = args.metrics if args.metrics else DEFAULT_PATTERNS
    summaries = [summarize_one(path) for path in expand_inputs(inputs)]
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
