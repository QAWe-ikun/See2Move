from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def summarize(path: Path) -> Dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    weak_actions = {
        action: metrics
        for action, metrics in report.get("per_action", {}).items()
        if float(metrics.get("f1", 0.0)) < 0.25
    }
    over_predicted = {
        action: {
            "support": metrics.get("support"),
            "predicted": metrics.get("predicted"),
            "ratio": metrics.get("predicted", 0) / max(1, metrics.get("support", 0)),
        }
        for action, metrics in report.get("per_action", {}).items()
        if metrics.get("predicted", 0) > metrics.get("support", 0) * 1.2
    }
    under_predicted = {
        action: {
            "support": metrics.get("support"),
            "predicted": metrics.get("predicted"),
            "ratio": metrics.get("predicted", 0) / max(1, metrics.get("support", 0)),
        }
        for action, metrics in report.get("per_action", {}).items()
        if metrics.get("predicted", 0) < metrics.get("support", 0) * 0.5
    }
    return {
        "path": str(path),
        "macro_precision": report.get("macro_precision"),
        "macro_recall": report.get("macro_recall"),
        "macro_f1": report.get("macro_f1"),
        "weak_actions": weak_actions,
        "over_predicted": over_predicted,
        "under_predicted": under_predicted,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("analysis", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.analysis), indent=2))


if __name__ == "__main__":
    main()
