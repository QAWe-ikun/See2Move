from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import yaml

from see2move.data.ai2thor_records import (
    build_action_vocab,
    candidate_score_features,
    load_depth,
    pose_features,
    read_jsonl,
    split_records,
    split_records_by_scene,
    use_modality,
)
from see2move.models.policy import build_qwen_model
from see2move.training.train_policy import (
    build_class_weights,
    build_balanced_sampler,
    classification_metrics,
    empty_oracle_score_metrics,
    evaluate,
    gain_score_loss,
    move_batch,
    oracle_soft_target_loss,
    set_seed,
    update_class_counts,
    update_oracle_score_metrics,
    update_topk_counts,
    metric_value,
    supervised_classification_loss,
)


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_qwen_feature_payload(path: Path) -> Dict[str, Any]:
    import torch

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload.get("features"), torch.Tensor):
        payload["features"] = torch.stack(payload["features"], dim=0)
    payload["features"] = payload["features"].contiguous()
    payload["id_to_index"] = {
        int(record_id): idx for idx, record_id in enumerate(payload["record_ids"])
    }
    return payload


def depth_feature_stack(path: Path, image_size: int, max_depth: float) -> np.ndarray:
    depth = load_depth(path, image_size=image_size, max_depth=max_depth)[0]
    inverse_depth = 1.0 - depth
    edge = np.zeros_like(depth)
    edge[:, 1:] += np.abs(depth[:, 1:] - depth[:, :-1])
    edge[1:, :] += np.abs(depth[1:, :] - depth[:-1, :])
    edge = np.clip(edge * 4.0, 0.0, 1.0)
    return np.stack([depth, inverse_depth, edge], axis=0).astype(np.float32)


class QwenFeatureOracleDataset:
    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        data_dir: Path,
        qwen_features: Any,
        id_to_feature_index: Dict[int, int],
        action_vocab: Dict[str, int],
        image_size: int,
        max_depth: float,
        ablation: Dict[str, Any] | None = None,
    ) -> None:
        self.records = list(records)
        self.data_dir = data_dir
        self.qwen_features = qwen_features
        self.id_to_feature_index = id_to_feature_index
        self.action_vocab = action_vocab
        self.image_size = image_size
        self.max_depth = max_depth
        self.ablation = ablation or {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        import torch

        record = self.records[index]
        record_id = int(record.get("id", index))
        feature_index = self.id_to_feature_index[record_id]
        qwen_feature = self.qwen_features[feature_index]
        depth = depth_feature_stack(
            self.data_dir / record["depth"],
            image_size=self.image_size,
            max_depth=self.max_depth,
        )
        pose = pose_features(record["agent"])
        if not use_modality(self.ablation, "qwen"):
            qwen_feature = torch.zeros_like(qwen_feature)
        if not use_modality(self.ablation, "depth"):
            depth = np.zeros_like(depth)
        if not use_modality(self.ablation, "pose"):
            pose = np.zeros_like(pose)
        label = self.action_vocab[record["label"]]
        candidate_scores, candidate_mask = candidate_score_features(record, self.action_vocab)
        return {
            "record_id": torch.tensor(record_id, dtype=torch.long),
            "qwen_feature": qwen_feature,
            "depth": torch.from_numpy(depth),
            "pose": torch.from_numpy(pose),
            "candidate_scores": torch.from_numpy(candidate_scores),
            "candidate_mask": torch.from_numpy(candidate_mask),
            "initial_visible_pixels": torch.tensor(
                float(record.get("initial_visible_pixels", 0.0)),
                dtype=torch.float32,
            ),
            "label": torch.tensor(label, dtype=torch.long),
        }


def split_for_training(records: Sequence[Dict[str, Any]], config: Dict[str, Any]) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]], Dict[str, Any]]:
    seed = int(config.get("seed", 7))
    data_cfg = config["data"]
    split_strategy = str(data_cfg.get("split_strategy", "scene"))
    val_fraction = float(data_cfg.get("val_fraction", 0.2))
    if split_strategy == "scene":
        train_records, val_records, val_scenes = split_records_by_scene(
            records,
            val_fraction=val_fraction,
            seed=seed,
        )
        split_info = {
            "strategy": split_strategy,
            "train_count": len(train_records),
            "val_count": len(val_records),
            "val_scenes": val_scenes,
        }
    elif split_strategy == "random":
        train_records, val_records = split_records(records, val_fraction=val_fraction, seed=seed)
        split_info = {
            "strategy": split_strategy,
            "train_count": len(train_records),
            "val_count": len(val_records),
        }
    else:
        raise ValueError(f"Unsupported split strategy: {split_strategy}")
    return train_records, val_records, split_info


def train(config: Dict[str, Any]) -> None:
    import torch
    from torch.utils.data import DataLoader

    seed = int(config.get("seed", 7))
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)

    data_cfg = config["data"]
    train_cfg = config["train"]
    model_cfg = config.get("model", {})
    data_dir = Path(data_cfg["data_dir"])
    records = read_jsonl(Path(data_cfg["records_path"]))
    if not records:
        raise ValueError("No records found. Generate AI2-THOR data first.")

    feature_path = data_cfg.get("feature_path") or data_cfg.get("features_path") or data_cfg["qwen_features_path"]
    feature_payload = load_qwen_feature_payload(Path(feature_path))
    action_vocab = build_action_vocab(records, config.get("candidate_actions"))
    idx_to_action = {idx: action for action, idx in action_vocab.items()}
    train_records, val_records, split_info = split_for_training(records, config)

    dataset_kwargs = {
        "data_dir": data_dir,
        "qwen_features": feature_payload["features"],
        "id_to_feature_index": feature_payload["id_to_index"],
        "action_vocab": action_vocab,
        "image_size": int(data_cfg.get("image_size", 128)),
        "max_depth": float(data_cfg.get("max_depth", 5.0)),
        "ablation": data_cfg.get("ablation", config.get("ablation", {})),
    }
    train_dataset = QwenFeatureOracleDataset(train_records, **dataset_kwargs)
    val_dataset = QwenFeatureOracleDataset(val_records, **dataset_kwargs) if val_records else None

    device = str(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    qwen_dim = int(feature_payload["features"].shape[1])
    model = build_qwen_model(config, qwen_dim=qwen_dim, num_actions=len(action_vocab)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    scheduler_name = str(train_cfg.get("scheduler", "none"))
    scheduler = None
    if scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(train_cfg.get("epochs", 10)),
            eta_min=float(train_cfg.get("min_learning_rate", 1e-5)),
        )
    elif scheduler_name != "none":
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    class_weighting = str(train_cfg.get("class_weighting", "none"))
    class_weights = None
    if class_weighting == "inverse_frequency":
        max_class_weight = train_cfg.get("max_class_weight")
        class_weights = build_class_weights(
            train_records,
            action_vocab,
            float(max_class_weight) if max_class_weight is not None else None,
            device,
        )
    elif class_weighting != "none":
        raise ValueError(f"Unsupported class weighting: {class_weighting}")
    focal_gamma = float(train_cfg.get("focal_loss_gamma", 0.0))
    soft_target_weight = float(train_cfg.get("soft_target_weight", 0.0))
    soft_target_temperature = float(train_cfg.get("soft_target_temperature", 2000.0))
    objective = str(train_cfg.get("objective", "classification"))
    score_scale = float(train_cfg.get("score_scale", 1000.0))
    score_loss_weight = float(train_cfg.get("score_loss_weight", 1.0))
    ranking_loss_weight = float(train_cfg.get("ranking_loss_weight", 0.5))
    ranking_margin = float(train_cfg.get("ranking_margin", 1.0))
    ce_loss_weight = float(train_cfg.get("ce_loss_weight", 0.1))
    if objective not in {"classification", "gain_score"}:
        raise ValueError(f"Unsupported objective: {objective}")
    gradient_clip_norm = train_cfg.get("gradient_clip_norm")
    gradient_clip_norm = float(gradient_clip_norm) if gradient_clip_norm is not None else None

    sampler_name = str(train_cfg.get("sampler", "none"))
    sampler = None
    if sampler_name == "class_balanced":
        sampler = build_balanced_sampler(train_records, action_vocab)
    elif sampler_name != "none":
        raise ValueError(f"Unsupported sampler: {sampler_name}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg.get("batch_size", 64)),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=int(train_cfg.get("batch_size", 64)),
            shuffle=False,
            num_workers=int(train_cfg.get("num_workers", 0)),
        )
        if val_dataset is not None
        else None
    )

    output_dir = Path(train_cfg.get("output_dir", "runs/ai2thor_qwen_policy"))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"split": split_info}))
    if class_weights is not None:
        print(
            json.dumps(
                {
                    "loss": {
                        "class_weighting": class_weighting,
                        "class_weights": {
                            idx_to_action[idx]: float(class_weights[idx].detach().cpu().item())
                            for idx in range(len(idx_to_action))
                        },
                    }
                }
            )
        )
    if sampler is not None or focal_gamma > 0:
        print(
            json.dumps(
                {
                    "imbalance": {
                        "sampler": sampler_name,
                        "focal_loss_gamma": focal_gamma,
                    }
                }
            )
        )
    if objective == "gain_score":
        print(
            json.dumps(
                {
                    "objective": {
                        "name": objective,
                        "score_scale": score_scale,
                        "score_loss_weight": score_loss_weight,
                        "ranking_loss_weight": ranking_loss_weight,
                        "ranking_margin": ranking_margin,
                        "ce_loss_weight": ce_loss_weight,
                    }
                }
            )
        )

    metrics = []
    selection_metric = str(train_cfg.get("selection_metric", "accuracy"))
    best_score = -1.0
    best_path = output_dir / "checkpoint_best.pt"
    patience = train_cfg.get("early_stopping_patience")
    patience = int(patience) if patience is not None else None
    min_delta = float(train_cfg.get("early_stopping_min_delta", 0.0))
    epochs_without_improvement = 0

    for epoch in range(1, int(train_cfg.get("epochs", 10)) + 1):
        model.train()
        total = 0
        correct = 0
        loss_sum = 0.0
        ce_loss_sum = 0.0
        soft_loss_sum = 0.0
        score_loss_sum = 0.0
        ranking_loss_sum = 0.0
        correct_by_class = [0 for _ in range(len(action_vocab))]
        total_by_class = [0 for _ in range(len(action_vocab))]
        predicted_by_class = [0 for _ in range(len(action_vocab))]
        topk_correct = {2: 0, 3: 0}
        score_metrics = empty_oracle_score_metrics()

        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            if objective == "gain_score":
                loss, loss_parts = gain_score_loss(
                    logits,
                    batch,
                    score_scale=score_scale,
                    score_loss_weight=score_loss_weight,
                    ranking_loss_weight=ranking_loss_weight,
                    ranking_margin=ranking_margin,
                    ce_loss_weight=ce_loss_weight,
                    class_weights=class_weights,
                    focal_gamma=focal_gamma,
                )
                ce_loss_value = loss_parts["ce_loss"]
                soft_loss_value = 0.0
                score_loss_value = loss_parts["score_regression_loss"]
                ranking_loss_value = loss_parts["ranking_loss"]
            else:
                ce_loss = supervised_classification_loss(
                    logits,
                    batch["label"],
                    class_weights,
                    focal_gamma,
                )
                soft_loss = logits.new_tensor(0.0)
                if soft_target_weight > 0:
                    soft_loss = oracle_soft_target_loss(
                        logits,
                        batch,
                        temperature=soft_target_temperature,
                    )
                loss = ce_loss + soft_target_weight * soft_loss
                ce_loss_value = float(ce_loss.item())
                soft_loss_value = float(soft_loss.item())
                score_loss_value = 0.0
                ranking_loss_value = 0.0
            loss.backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()

            batch_count = int(batch["label"].numel())
            loss_sum += float(loss.item()) * batch_count
            ce_loss_sum += ce_loss_value * batch_count
            soft_loss_sum += soft_loss_value * batch_count
            score_loss_sum += score_loss_value * batch_count
            ranking_loss_sum += ranking_loss_value * batch_count
            pred = logits.argmax(dim=1)
            correct += int((pred == batch["label"]).sum().item())
            total += batch_count
            update_class_counts(pred, batch["label"], correct_by_class, total_by_class, predicted_by_class)
            update_topk_counts(logits, batch["label"], topk_correct)
            update_oracle_score_metrics(pred, batch, score_metrics)

        train_metrics = classification_metrics(
            loss_sum,
            correct,
            total,
            correct_by_class,
            total_by_class,
            idx_to_action,
            predicted_by_class,
            topk_correct,
            score_metrics,
        )
        train_metrics["ce_loss"] = ce_loss_sum / max(1, total)
        if objective == "gain_score":
            train_metrics["objective"] = objective
            train_metrics["score_regression_loss"] = score_loss_sum / max(1, total)
            train_metrics["ranking_loss"] = ranking_loss_sum / max(1, total)
        if soft_target_weight > 0:
            train_metrics["soft_target_loss"] = soft_loss_sum / max(1, total)
        val_metrics = (
            evaluate(
                model,
                val_loader,
                device,
                idx_to_action,
                objective=objective,
                score_scale=score_scale,
                score_loss_weight=score_loss_weight,
                ranking_loss_weight=ranking_loss_weight,
                ranking_margin=ranking_margin,
                ce_loss_weight=ce_loss_weight,
            )
            if val_loader is not None
            else {}
        )
        if scheduler is not None:
            scheduler.step()

        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        metrics.append(row)
        print(json.dumps(row))

        score_source = val_metrics if val_metrics else train_metrics
        score = metric_value(score_source, selection_metric, metric_value(score_source, "accuracy", 0.0))
        if score > best_score + min_delta:
            best_score = score
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "action_vocab": action_vocab,
                    "qwen_feature_dim": qwen_dim,
                    "split": split_info,
                    "selection_metric": selection_metric,
                    "epoch": epoch,
                    "metrics": row,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1
            if patience is not None and epochs_without_improvement >= patience:
                print(
                    json.dumps(
                        {
                            "early_stop": {
                                "epoch": epoch,
                                "selection_metric": selection_metric,
                                "best_score": best_score,
                                "patience": patience,
                            }
                        }
                    )
                )
                break

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved best checkpoint to {best_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/train_ai2thor_qwen3vl.yaml"))
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
