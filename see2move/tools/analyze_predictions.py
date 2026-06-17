from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def confusion_report(labels: List[int], preds: List[int], idx_to_action: Dict[int, str]) -> Dict[str, Any]:
    num_actions = len(idx_to_action)
    matrix = [[0 for _ in range(num_actions)] for _ in range(num_actions)]
    for label, pred in zip(labels, preds):
        matrix[label][pred] += 1

    per_action = {}
    f1_scores = []
    recalls = []
    precisions = []
    for idx in range(num_actions):
        tp = matrix[idx][idx]
        support = sum(matrix[idx])
        predicted = sum(row[idx] for row in matrix)
        precision = tp / max(1, predicted)
        recall = tp / max(1, support)
        f1 = 2.0 * precision * recall / max(1.0e-8, precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        per_action[idx_to_action[idx]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
            "predicted": predicted,
        }

    return {
        "actions": [idx_to_action[idx] for idx in range(num_actions)],
        "confusion_matrix": matrix,
        "macro_precision": sum(precisions) / max(1, len(precisions)),
        "macro_recall": sum(recalls) / max(1, len(recalls)),
        "macro_f1": sum(f1_scores) / max(1, len(f1_scores)),
        "per_action": per_action,
    }


def collect_predictions(
    model: Any,
    loader: Any,
    device: str,
    idx_to_action: Dict[int, str],
    objective: str,
    score_scale: float,
    stay_threshold: float | None,
    stay_ratio_threshold: float | None,
    stay_ratio_epsilon: float,
) -> tuple[List[int], List[int], List[int], List[float]]:
    import torch

    from see2move.training.train_policy import select_predictions

    model.eval()
    labels: List[int] = []
    preds: List[int] = []
    record_ids: List[int] = []
    predicted_scores: List[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(batch)
            pred = select_predictions(
                logits,
                batch,
                idx_to_action,
                objective=objective,
                score_scale=score_scale,
                stay_threshold=stay_threshold,
                stay_ratio_threshold=stay_ratio_threshold,
                stay_ratio_epsilon=stay_ratio_epsilon,
            )
            preds.extend(pred.detach().cpu().tolist())
            labels.extend(batch["label"].detach().cpu().tolist())
            if "record_id" in batch:
                record_ids.extend(batch["record_id"].detach().cpu().tolist())
            if "candidate_scores" in batch:
                scores = batch["candidate_scores"].detach().cpu()
                pred_cpu = pred.detach().cpu().long()
                predicted_scores.extend(scores.gather(1, pred_cpu.unsqueeze(1)).squeeze(1).tolist())
    return labels, preds, record_ids, predicted_scores


def attach_record_predictions(
    report: Dict[str, Any],
    records: List[Dict[str, Any]],
    labels: List[int],
    preds: List[int],
    record_ids: List[int],
    predicted_scores: List[float],
    idx_to_action: Dict[int, str],
) -> None:
    records_by_id = {int(record.get("id", idx)): record for idx, record in enumerate(records)}
    rows = []
    for offset, (label, pred) in enumerate(zip(labels, preds)):
        record_id = int(record_ids[offset]) if offset < len(record_ids) else offset
        record = records_by_id.get(record_id, {})
        candidates = record.get("candidates", [])
        best_score = max((float(candidate.get("score", 0.0)) for candidate in candidates), default=0.0)
        predicted_score = (
            float(predicted_scores[offset]) if offset < len(predicted_scores) else 0.0
        )
        initial_visible = float(record.get("initial_visible_pixels", 0.0))
        rows.append(
            {
                "id": record_id,
                "scene": record.get("scene"),
                "object_type": record.get("target", {}).get("object_type"),
                "initial_visible_pixels": initial_visible,
                "best_score": best_score,
                "label": idx_to_action[label],
                "prediction": idx_to_action[pred],
                "correct": int(label == pred),
                "predicted_score": predicted_score,
                "positive_gain": int(predicted_score > 0.0),
                "predicted_relative_gain": predicted_score / max(1.0, initial_visible),
                "best_relative_gain": best_score / max(1.0, initial_visible),
            }
        )
    report["records"] = rows


def analyze_simple(args: argparse.Namespace) -> Dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    from see2move.data.ai2thor_records import AI2ThorOracleDataset, read_jsonl
    from see2move.models.policy import build_model

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt["config"]
    data_cfg = config["data"]
    records = read_jsonl(args.records)
    dataset = AI2ThorOracleDataset(
        records,
        data_dir=args.data_dir,
        text_vocab=ckpt["text_vocab"],
        action_vocab=ckpt["action_vocab"],
        image_size=int(data_cfg.get("image_size", 128)),
        max_depth=float(data_cfg.get("max_depth", 5.0)),
        max_text_len=int(data_cfg.get("max_text_len", 32)),
        ablation=data_cfg.get("ablation", config.get("ablation", {})),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size)
    model = build_model(config, len(ckpt["text_vocab"]), len(ckpt["action_vocab"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(args.device)
    idx_to_action = {idx: action for action, idx in ckpt["action_vocab"].items()}
    train_cfg = config.get("train", {})
    labels, preds, record_ids, predicted_scores = collect_predictions(
        model,
        loader,
        args.device,
        idx_to_action,
        objective=str(train_cfg.get("objective", "classification")),
        score_scale=float(train_cfg.get("score_scale", 1000.0)),
        stay_threshold=args.stay_threshold,
        stay_ratio_threshold=args.stay_ratio_threshold,
        stay_ratio_epsilon=args.stay_ratio_epsilon,
    )
    report = confusion_report(labels, preds, idx_to_action)
    if args.include_records:
        attach_record_predictions(
            report,
            records,
            labels,
            preds,
            record_ids,
            predicted_scores,
            idx_to_action,
        )
    return report


def analyze_qwen(args: argparse.Namespace) -> Dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader

    from see2move.data.ai2thor_records import build_action_vocab, read_jsonl
    from see2move.models.policy import build_qwen_model
    from see2move.training.train_qwen_policy import QwenFeatureOracleDataset, load_qwen_feature_payload

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt["config"]
    data_cfg = config["data"]
    records = read_jsonl(args.records)
    action_vocab = ckpt.get("action_vocab") or build_action_vocab(records, config.get("candidate_actions"))
    features = load_qwen_feature_payload(args.qwen_features)
    dataset = QwenFeatureOracleDataset(
        records,
        data_dir=args.data_dir,
        qwen_features=features["features"],
        id_to_feature_index=features["id_to_index"],
        action_vocab=action_vocab,
        image_size=int(data_cfg.get("image_size", 128)),
        max_depth=float(data_cfg.get("max_depth", 5.0)),
        ablation=data_cfg.get("ablation", config.get("ablation", {})),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size)
    model = build_qwen_model(
        config,
        qwen_dim=int(ckpt.get("qwen_feature_dim", features["features"].shape[1])),
        num_actions=len(action_vocab),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(args.device)
    idx_to_action = {idx: action for action, idx in action_vocab.items()}
    train_cfg = config.get("train", {})
    labels, preds, record_ids, predicted_scores = collect_predictions(
        model,
        loader,
        args.device,
        idx_to_action,
        objective=str(train_cfg.get("objective", "classification")),
        score_scale=float(train_cfg.get("score_scale", 1000.0)),
        stay_threshold=args.stay_threshold,
        stay_ratio_threshold=args.stay_ratio_threshold,
        stay_ratio_epsilon=args.stay_ratio_epsilon,
    )
    report = confusion_report(labels, preds, idx_to_action)
    if args.include_records:
        attach_record_predictions(
            report,
            records,
            labels,
            preds,
            record_ids,
            predicted_scores,
            idx_to_action,
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--kind", choices=["auto", "simple", "qwen"], default="auto")
    parser.add_argument("--records", type=Path, default=Path("/mnt/f/see2move/data/ai2thor_oracle_10k_stay/records.jsonl"))
    parser.add_argument("--data-dir", type=Path, default=Path("/mnt/f/see2move/data/ai2thor_oracle_10k_stay"))
    parser.add_argument("--qwen-features", type=Path, default=Path("/mnt/f/see2move/features/qwen3vl_ai2thor_oracle_10k_stay.pt"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-records", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stay-threshold", type=float, default=None)
    parser.add_argument("--stay-ratio-threshold", type=float, default=None)
    parser.add_argument("--stay-ratio-epsilon", type=float, default=1.0)
    args = parser.parse_args()

    import torch

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.kind == "auto":
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        args.kind = "qwen" if "qwen_feature_dim" in ckpt else "simple"

    report = analyze_qwen(args) if args.kind == "qwen" else analyze_simple(args)
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
