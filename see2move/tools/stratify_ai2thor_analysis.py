from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List


def read_records(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def best_score(record: Dict[str, Any]) -> float:
    return max(
        (float(candidate.get("score", 0.0)) for candidate in record.get("candidates", [])),
        default=0.0,
    )


def visibility_bin(value: float) -> str:
    if value <= 1000:
        return "low_visible_<=1k"
    if value <= 5000:
        return "mid_visible_1k_5k"
    return "high_visible_>5k"


def best_gain_bin(value: float) -> str:
    if value <= 0:
        return "no_oracle_gain"
    if value <= 1000:
        return "small_gain_0_1k"
    if value <= 5000:
        return "medium_gain_1k_5k"
    return "large_gain_>5k"


def relative_gain_bin(value: float) -> str:
    if value <= 0:
        return "relative_no_gain"
    if value <= 0.25:
        return "relative_small_0_25"
    if value <= 1.0:
        return "relative_mid_25_100"
    return "relative_large_>100"


def add_dataset_row(groups: Dict[str, List[Dict[str, Any]]], key: str, record: Dict[str, Any]) -> None:
    groups[key].append(record)


def dataset_group_stats(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    labels = Counter(str(record.get("label")) for record in rows)
    best_scores = [best_score(record) for record in rows]
    initial_visible = [float(record.get("initial_visible_pixels", 0.0)) for record in rows]
    return {
        "count": len(rows),
        "majority_label": labels.most_common(1)[0][0] if labels else None,
        "majority_rate": labels.most_common(1)[0][1] / max(1, len(rows)) if labels else 0.0,
        "mean_initial_visible_pixels": mean(initial_visible) if initial_visible else 0.0,
        "mean_best_score": mean(best_scores) if best_scores else 0.0,
        "best_positive_gain_rate": sum(1 for score in best_scores if score > 0) / max(1, len(best_scores)),
        "labels": dict(labels.most_common()),
    }


def prediction_group_stats(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    items = list(rows)
    predicted_scores = [float(item.get("predicted_score", 0.0)) for item in items]
    best_scores = [float(item.get("best_score", 0.0)) for item in items]
    return {
        "count": len(items),
        "accuracy": sum(int(item.get("correct", 0)) for item in items) / max(1, len(items)),
        "positive_gain_rate": sum(int(item.get("positive_gain", 0)) for item in items) / max(1, len(items)),
        "mean_predicted_score": mean(predicted_scores) if predicted_scores else 0.0,
        "mean_best_score": mean(best_scores) if best_scores else 0.0,
        "mean_predicted_relative_gain": mean(
            float(item.get("predicted_relative_gain", 0.0)) for item in items
        )
        if items
        else 0.0,
    }


def summarize_dataset(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        initial_visible = float(record.get("initial_visible_pixels", 0.0))
        gain = best_score(record)
        relative_gain = gain / max(1.0, initial_visible)
        add_dataset_row(groups, f"initial_visibility/{visibility_bin(initial_visible)}", record)
        add_dataset_row(groups, f"oracle_gain/{best_gain_bin(gain)}", record)
        add_dataset_row(groups, f"relative_oracle_gain/{relative_gain_bin(relative_gain)}", record)
        add_dataset_row(groups, f"label/{record.get('label')}", record)
        add_dataset_row(groups, f"object/{record.get('target', {}).get('object_type')}", record)
    return {key: dataset_group_stats(value) for key, value in sorted(groups.items())}


def summarize_predictions(prediction_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        initial_visible = float(row.get("initial_visible_pixels", 0.0))
        gain = float(row.get("best_score", 0.0))
        relative_gain = float(row.get("best_relative_gain", 0.0))
        groups[f"initial_visibility/{visibility_bin(initial_visible)}"].append(row)
        groups[f"oracle_gain/{best_gain_bin(gain)}"].append(row)
        groups[f"relative_oracle_gain/{relative_gain_bin(relative_gain)}"].append(row)
        groups[f"label/{row.get('label')}"].append(row)
        groups[f"object/{row.get('object_type')}"].append(row)
    return {key: prediction_group_stats(value) for key, value in sorted(groups.items())}


def summarize(records_path: Path, analysis_path: Path | None) -> Dict[str, Any]:
    records = read_records(records_path)
    result = {
        "records_path": str(records_path),
        "count": len(records),
        "dataset_strata": summarize_dataset(records),
    }
    if analysis_path is not None:
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
        prediction_rows = analysis.get("records", [])
        if not prediction_rows:
            result["prediction_warning"] = (
                "analysis file has no per-record predictions; rerun analyze_ai2thor_predictions "
                "with --include-records"
            )
        else:
            result["analysis_path"] = str(analysis_path)
            result["prediction_strata"] = summarize_predictions(prediction_rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=Path("/mnt/f/see2move/data/ai2thor_oracle_10k_stay/records.jsonl"))
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = summarize(args.records, args.analysis)
    text = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
