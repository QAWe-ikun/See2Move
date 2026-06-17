from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

def candidate_actions(record: Dict[str, Any], action_vocab: Dict[str, int]) -> List[str]:
    actions = [
        str(candidate.get("action"))
        for candidate in record.get("candidates", [])
        if str(candidate.get("action")) in action_vocab
    ]
    return actions or list(action_vocab)


def choose_best_score_action(record: Dict[str, Any], action_vocab: Dict[str, int]) -> str:
    best_action = None
    best_score = -float("inf")
    for candidate in record.get("candidates", []):
        action = str(candidate.get("action"))
        if action not in action_vocab:
            continue
        score = float(candidate.get("score", 0.0))
        if score > best_score:
            best_action = action
            best_score = score
    return best_action or next(iter(action_vocab))


def choose_best_moving_or_stay(
    record: Dict[str, Any],
    action_vocab: Dict[str, int],
    stay_threshold: float,
) -> str:
    best_action = None
    best_score = -float("inf")
    for candidate in record.get("candidates", []):
        action = str(candidate.get("action"))
        if action not in action_vocab or action == "Stay":
            continue
        score = float(candidate.get("score", 0.0))
        if score > best_score:
            best_action = action
            best_score = score
    if "Stay" in action_vocab and best_score <= stay_threshold:
        return "Stay"
    return best_action or ("Stay" if "Stay" in action_vocab else next(iter(action_vocab)))


def choose_positive_oracle(record: Dict[str, Any], action_vocab: Dict[str, int]) -> str:
    return choose_best_moving_or_stay(record, action_vocab, stay_threshold=0.0)


def evaluate_predictions(
    records: Sequence[Dict[str, Any]],
    action_vocab: Dict[str, int],
    predictions: Sequence[str],
) -> Dict[str, Any]:
    import torch

    from see2move.data.ai2thor_records import candidate_score_features
    from see2move.training.train_policy import (
        classification_metrics,
        empty_oracle_score_metrics,
        update_class_counts,
        update_oracle_score_metrics,
    )

    idx_to_action = {idx: action for action, idx in action_vocab.items()}
    labels = torch.tensor([action_vocab[record["label"]] for record in records], dtype=torch.long)
    preds = torch.tensor([action_vocab[action] for action in predictions], dtype=torch.long)
    num_actions = len(action_vocab)
    correct_by_class = [0 for _ in range(num_actions)]
    total_by_class = [0 for _ in range(num_actions)]
    predicted_by_class = [0 for _ in range(num_actions)]
    score_metrics = empty_oracle_score_metrics()
    update_class_counts(preds, labels, correct_by_class, total_by_class, predicted_by_class)

    for record, pred_action in zip(records, predictions):
        candidate_scores, candidate_mask = candidate_score_features(record, action_vocab)
        batch = {
            "candidate_scores": torch.from_numpy(candidate_scores).unsqueeze(0),
            "candidate_mask": torch.from_numpy(candidate_mask).unsqueeze(0),
            "initial_visible_pixels": torch.tensor(
                [float(record.get("initial_visible_pixels", 0.0))],
                dtype=torch.float32,
            ),
        }
        update_oracle_score_metrics(torch.tensor([action_vocab[pred_action]]), batch, score_metrics)

    return classification_metrics(
        0.0,
        int((preds == labels).sum().item()),
        len(records),
        correct_by_class,
        total_by_class,
        idx_to_action,
        predicted_by_class=predicted_by_class,
        topk_correct=None,
        score_metrics=score_metrics,
    )


def build_predictions(
    name: str,
    records: Sequence[Dict[str, Any]],
    action_vocab: Dict[str, int],
    rng: random.Random,
    stay_threshold: float,
) -> List[str]:
    label_counts = Counter(str(record["label"]) for record in records)
    majority = label_counts.most_common(1)[0][0]
    if name == "random":
        return [rng.choice(candidate_actions(record, action_vocab)) for record in records]
    if name == "majority":
        return [majority for _ in records]
    if name == "stay":
        stay = "Stay" if "Stay" in action_vocab else majority
        return [stay for _ in records]
    if name == "oracle":
        return [choose_best_score_action(record, action_vocab) for record in records]
    if name == "positive_oracle":
        return [choose_positive_oracle(record, action_vocab) for record in records]
    if name == "threshold_oracle":
        return [choose_best_moving_or_stay(record, action_vocab, stay_threshold) for record in records]
    raise ValueError(f"Unknown baseline: {name}")


def markdown_table(rows: Iterable[Dict[str, Any]]) -> str:
    columns = [
        ("Baseline", "baseline"),
        ("Acc", "accuracy"),
        ("Macro F1", "macro_f1"),
        ("Mean Gain", "mean_predicted_score"),
        ("Rel Gain", "mean_predicted_relative_gain"),
        ("Pos Gain", "positive_gain_rate"),
        ("Neg Gain", "negative_gain_rate"),
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


def flatten_metrics(name: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    stay = metrics.get("per_action", {}).get("Stay", {})
    gain = metrics.get("oracle_gain", {})
    return {
        "baseline": name,
        "accuracy": metrics.get("accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "mean_predicted_score": gain.get("mean_predicted_score"),
        "mean_best_score": gain.get("mean_best_score"),
        "mean_predicted_relative_gain": gain.get("mean_predicted_relative_gain"),
        "mean_best_relative_gain": gain.get("mean_best_relative_gain"),
        "positive_gain_rate": gain.get("positive_gain_rate"),
        "nonnegative_gain_rate": gain.get("nonnegative_gain_rate"),
        "negative_gain_rate": gain.get("negative_gain_rate"),
        "stay_precision": stay.get("precision"),
        "stay_recall": stay.get("recall"),
        "stay_predicted_count": stay.get("predicted_count"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=Path("/mnt/f/see2move/data/ai2thor_oracle_10k_stay/records.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/baselines_10k_stay"))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stay-threshold", type=float, default=200.0)
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["random", "majority", "stay", "positive_oracle", "oracle", "threshold_oracle"],
    )
    args = parser.parse_args()

    from see2move.data.ai2thor_records import build_action_vocab, read_jsonl

    records = read_jsonl(args.records)
    action_vocab = build_action_vocab(records)
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    detailed: Dict[str, Any] = {}
    for baseline in args.baselines:
        predictions = build_predictions(baseline, records, action_vocab, rng, args.stay_threshold)
        metrics = evaluate_predictions(records, action_vocab, predictions)
        detailed[baseline] = metrics
        rows.append(flatten_metrics(baseline, metrics))

    (args.output_dir / "baselines.json").write_text(json.dumps(detailed, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "summary.md").write_text(markdown_table(rows), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
