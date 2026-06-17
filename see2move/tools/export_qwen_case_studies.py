from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List

def save_depth_preview(depth_path: Path, output_path: Path) -> None:
    import numpy as np
    from PIL import Image

    depth = np.load(depth_path).astype(np.float32)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    high = float(np.percentile(depth, 95)) if depth.size else 1.0
    if high <= 0:
        high = float(depth.max()) if depth.size else 1.0
    high = max(high, 1.0e-6)
    vis = np.clip(depth / high, 0.0, 1.0)
    Image.fromarray((vis * 255.0).astype(np.uint8)).save(output_path)


def candidate_true_scores(record: Dict[str, Any]) -> Dict[str, float]:
    return {
        str(candidate.get("action")): float(candidate.get("score", 0.0))
        for candidate in record.get("candidates", [])
    }


def collect_model_rows(args: argparse.Namespace) -> List[Dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader

    from see2move.data.ai2thor_records import build_action_vocab, read_jsonl
    from see2move.models.policy import build_qwen_model
    from see2move.training.train_policy import select_predictions
    from see2move.training.train_qwen_policy import QwenFeatureOracleDataset, load_qwen_feature_payload

    ckpt: Dict[str, Any] = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt["config"]
    data_cfg = config["data"]
    train_cfg = config.get("train", {})
    records = read_jsonl(args.records)
    action_vocab = ckpt.get("action_vocab") or build_action_vocab(records, config.get("candidate_actions"))
    idx_to_action = {idx: action for action, idx in action_vocab.items()}
    feature_payload = load_qwen_feature_payload(args.qwen_features)

    dataset = QwenFeatureOracleDataset(
        records,
        data_dir=args.data_dir,
        qwen_features=feature_payload["features"],
        id_to_feature_index=feature_payload["id_to_index"],
        action_vocab=action_vocab,
        image_size=int(data_cfg.get("image_size", 128)),
        max_depth=float(data_cfg.get("max_depth", 5.0)),
        ablation=data_cfg.get("ablation", config.get("ablation", {})),
    )
    loader = DataLoader(dataset, batch_size=args.batch_size)
    model = build_qwen_model(
        config,
        qwen_dim=int(ckpt.get("qwen_feature_dim", feature_payload["features"].shape[1])),
        num_actions=len(action_vocab),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(args.device)
    model.eval()

    records_by_id = {int(record.get("id", idx)): record for idx, record in enumerate(records)}
    rows: List[Dict[str, Any]] = []
    score_scale = float(train_cfg.get("score_scale", 1000.0))
    objective = str(train_cfg.get("objective", "classification"))

    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(args.device) for key, value in batch.items()}
            logits = model(batch)
            preds = select_predictions(
                logits,
                batch,
                idx_to_action,
                objective=objective,
                score_scale=score_scale,
                stay_threshold=args.stay_threshold,
                stay_ratio_threshold=args.stay_ratio_threshold,
                stay_ratio_epsilon=args.stay_ratio_epsilon,
            )
            top_values, top_indices = logits.topk(k=min(3, logits.shape[1]), dim=1)
            labels = batch["label"].detach().cpu().tolist()
            record_ids = batch["record_id"].detach().cpu().tolist()
            candidate_scores = batch["candidate_scores"].detach().cpu()
            initial_visible = batch["initial_visible_pixels"].detach().cpu().tolist()
            for offset, record_id in enumerate(record_ids):
                record = records_by_id[int(record_id)]
                pred_idx = int(preds[offset].detach().cpu().item())
                label_idx = int(labels[offset])
                pred_score = float(candidate_scores[offset, pred_idx].item())
                best_score = max(candidate_true_scores(record).values(), default=0.0)
                top3 = []
                for value, idx in zip(top_values[offset].detach().cpu().tolist(), top_indices[offset].detach().cpu().tolist()):
                    top3.append(
                        {
                            "action": idx_to_action[int(idx)],
                            "predicted_gain": float(value) * score_scale if objective == "gain_score" else float(value),
                            "true_gain": float(candidate_scores[offset, int(idx)].item()),
                        }
                    )
                rows.append(
                    {
                        "id": int(record_id),
                        "scene": record.get("scene"),
                        "instruction": record.get("instruction"),
                        "object_type": record.get("target", {}).get("object_type"),
                        "rgb": record.get("rgb"),
                        "depth": record.get("depth"),
                        "initial_visible_pixels": float(initial_visible[offset]),
                        "label": idx_to_action[label_idx],
                        "prediction": idx_to_action[pred_idx],
                        "correct": idx_to_action[label_idx] == idx_to_action[pred_idx],
                        "predicted_score": pred_score,
                        "best_score": best_score,
                        "predicted_relative_gain": pred_score / max(1.0, float(initial_visible[offset])),
                        "best_relative_gain": best_score / max(1.0, float(initial_visible[offset])),
                        "top3": top3,
                        "candidate_true_scores": candidate_true_scores(record),
                    }
                )
    return rows


def pick_cases(rows: List[Dict[str, Any]], per_group: int) -> List[Dict[str, Any]]:
    groups = [
        ("success_positive", lambda row: row["predicted_score"] > 0),
        ("failure_negative", lambda row: row["predicted_score"] < 0),
        ("stay_miss", lambda row: row["label"] == "Stay" and row["prediction"] != "Stay"),
        ("stay_hit", lambda row: row["label"] == "Stay" and row["prediction"] == "Stay"),
    ]
    picked: List[Dict[str, Any]] = []
    seen = set()
    for group, predicate in groups:
        candidates = [row for row in rows if predicate(row) and row["id"] not in seen]
        if group == "success_positive":
            candidates.sort(key=lambda row: row["predicted_score"], reverse=True)
        elif group == "failure_negative":
            candidates.sort(key=lambda row: row["predicted_score"])
        elif group == "stay_miss":
            candidates.sort(key=lambda row: row["best_score"], reverse=True)
        else:
            candidates.sort(key=lambda row: row["initial_visible_pixels"], reverse=True)
        for row in candidates[:per_group]:
            row = dict(row)
            row["case_group"] = group
            picked.append(row)
            seen.add(row["id"])
    return picked


def write_cases(cases: List[Dict[str, Any]], data_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# See2Move Case Studies", ""]
    for idx, case in enumerate(cases, start=1):
        case_dir = output_dir / f"{idx:02d}_{case['case_group']}_{case['id']}"
        case_dir.mkdir(parents=True, exist_ok=True)
        rgb_src = data_dir / str(case["rgb"])
        depth_src = data_dir / str(case["depth"])
        rgb_dst = case_dir / "initial_rgb.png"
        depth_dst = case_dir / "initial_depth.png"
        if rgb_src.exists():
            shutil.copyfile(rgb_src, rgb_dst)
        if depth_src.exists():
            save_depth_preview(depth_src, depth_dst)
        (case_dir / "metadata.json").write_text(json.dumps(case, indent=2) + "\n", encoding="utf-8")

        lines.extend(
            [
                f"## Case {idx}: {case['case_group']} / record {case['id']}",
                "",
                f"- Scene: `{case['scene']}`",
                f"- Instruction: {case['instruction']}",
                f"- Target: `{case['object_type']}`",
                f"- Label / Prediction: `{case['label']}` / `{case['prediction']}`",
                f"- True gain of prediction: {case['predicted_score']:.2f}",
                f"- Relative gain of prediction: {case['predicted_relative_gain']:.4f}",
                f"- Best true gain: {case['best_score']:.2f}",
                "",
                "| Rank | Predicted action | Predicted gain | True gain |",
                "| ---: | --- | ---: | ---: |",
            ]
        )
        for rank, item in enumerate(case["top3"], start=1):
            lines.append(
                f"| {rank} | {item['action']} | {item['predicted_gain']:.2f} | {item['true_gain']:.2f} |"
            )
        lines.extend(["", f"- RGB: `{rgb_dst}`", f"- Depth: `{depth_dst}`", ""])
    output_dir.joinpath("cases.json").write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
    output_dir.joinpath("cases.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/ai2thor_policy_10k_stay_qwen3vl_gain_score_e25/checkpoint_best.pt"))
    parser.add_argument("--records", type=Path, default=Path("/mnt/f/see2move/data/ai2thor_oracle_10k_stay/records.jsonl"))
    parser.add_argument("--data-dir", type=Path, default=Path("/mnt/f/see2move/data/ai2thor_oracle_10k_stay"))
    parser.add_argument("--qwen-features", type=Path, default=Path("/mnt/f/see2move/features/qwen3vl_ai2thor_oracle_10k_stay.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/case_studies_10k_stay_t200"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stay-threshold", type=float, default=200.0)
    parser.add_argument("--stay-ratio-threshold", type=float, default=None)
    parser.add_argument("--stay-ratio-epsilon", type=float, default=1.0)
    parser.add_argument("--per-group", type=int, default=2)
    args = parser.parse_args()

    import torch

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = collect_model_rows(args)
    cases = pick_cases(rows, per_group=args.per_group)
    write_cases(cases, args.data_dir, args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "case_count": len(cases)}, indent=2))


if __name__ == "__main__":
    main()
