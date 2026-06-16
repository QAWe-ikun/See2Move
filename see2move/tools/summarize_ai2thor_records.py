from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable


def count_values(records: Iterable[Dict[str, Any]], key: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        value: Any = record
        for part in key.split("."):
            if not isinstance(value, dict):
                value = None
                break
            value = value.get(part)
        counts[str(value)] += 1
    return counts


def read_records(path: Path) -> list[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def summarize(records_path: Path) -> Dict[str, Any]:
    records = read_records(records_path)
    label_counts = count_values(records, "label")
    scene_counts = count_values(records, "scene")
    object_counts = count_values(records, "target.object_type")
    action_count = len(label_counts)
    majority_count = label_counts.most_common(1)[0][1] if label_counts else 0

    return {
        "records_path": str(records_path),
        "count": len(records),
        "scene_count": len(scene_counts),
        "action_count": action_count,
        "random_action_accuracy": 1.0 / action_count if action_count else 0.0,
        "majority_label_accuracy": majority_count / max(1, len(records)),
        "labels": dict(label_counts.most_common()),
        "objects": dict(object_counts.most_common()),
        "top_scenes": dict(scene_counts.most_common(20)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/train_ai2thor_policy.yaml"))
    args = parser.parse_args()

    records_path = args.records
    if records_path is None:
        import yaml

        with args.config.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        records_path = Path(config["data"]["records_path"])

    print(json.dumps(summarize(records_path), indent=2))


if __name__ == "__main__":
    main()
