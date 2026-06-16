from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from see2move.data.ai2thor_records import (
    AI2ThorOracleDataset,
    build_action_vocab,
    build_text_vocab,
    read_jsonl,
    split_records,
    split_records_by_scene,
)
from see2move.models.policy import build_model


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def metric_value(metrics: Dict[str, Any], metric: str, default: float = 0.0) -> float:
    value: Any = metrics
    for part in metric.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    if isinstance(value, (int, float)):
        return float(value)
    return default


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    return {key: value.to(device) for key, value in batch.items()}


def update_class_counts(
    predictions: Any,
    labels: Any,
    correct_by_class: List[int],
    total_by_class: List[int],
    predicted_by_class: Optional[List[int]] = None,
) -> None:
    for pred, label in zip(predictions.detach().cpu().tolist(), labels.detach().cpu().tolist()):
        if predicted_by_class is not None:
            predicted_by_class[int(pred)] += 1
        total_by_class[int(label)] += 1
        if int(pred) == int(label):
            correct_by_class[int(label)] += 1


def update_topk_counts(
    logits: Any,
    labels: Any,
    topk_correct: Dict[int, int],
) -> None:
    max_k = max(topk_correct)
    top_indices = logits.detach().topk(k=min(max_k, logits.shape[1]), dim=1).indices.cpu()
    labels_cpu = labels.detach().cpu()
    for k in topk_correct:
        topk_correct[k] += int((top_indices[:, :k] == labels_cpu[:, None]).any(dim=1).sum().item())


def update_oracle_score_metrics(
    predictions: Any,
    batch: Dict[str, Any],
    score_metrics: Dict[str, float],
) -> None:
    if "candidate_scores" not in batch:
        return

    scores = batch["candidate_scores"].detach().cpu()
    mask = batch.get("candidate_mask")
    if mask is None:
        mask_cpu = scores.new_ones(scores.shape)
    else:
        mask_cpu = mask.detach().cpu()

    predictions_cpu = predictions.detach().cpu().long()
    predicted_scores = scores.gather(1, predictions_cpu.unsqueeze(1)).squeeze(1)
    masked_scores = scores.clone()
    masked_scores[mask_cpu <= 0] = -1.0e9
    best_scores = masked_scores.max(dim=1).values
    valid = best_scores > -1.0e8
    if not bool(valid.any()):
        return

    predicted_scores = predicted_scores[valid]
    best_scores = best_scores[valid]
    positive_best = best_scores > 0

    score_metrics["count"] += float(valid.sum().item())
    score_metrics["predicted_score_sum"] += float(predicted_scores.sum().item())
    score_metrics["best_score_sum"] += float(best_scores.sum().item())
    score_metrics["positive_gain_count"] += float((predicted_scores > 0).sum().item())
    score_metrics["nonnegative_gain_count"] += float((predicted_scores >= 0).sum().item())
    score_metrics["negative_gain_count"] += float((predicted_scores < 0).sum().item())
    score_metrics["best_positive_gain_count"] += float(positive_best.sum().item())
    if bool(positive_best.any()):
        ratio = predicted_scores[positive_best] / best_scores[positive_best].clamp_min(1.0)
        score_metrics["gain_ratio_sum"] += float(ratio.sum().item())
        score_metrics["gain_ratio_count"] += float(positive_best.sum().item())

    if "initial_visible_pixels" in batch:
        base_visible = batch["initial_visible_pixels"].detach().cpu()[valid].clamp_min(1.0)
        score_metrics["relative_predicted_gain_sum"] += float((predicted_scores / base_visible).sum().item())
        score_metrics["relative_best_gain_sum"] += float((best_scores / base_visible).sum().item())


def classification_metrics(
    loss_sum: float,
    correct: int,
    total: int,
    correct_by_class: List[int],
    total_by_class: List[int],
    idx_to_action: Dict[int, str],
    predicted_by_class: Optional[List[int]] = None,
    topk_correct: Optional[Dict[int, int]] = None,
    score_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    per_action: Dict[str, Dict[str, float]] = {}
    action_accuracies = []
    action_f1_scores = []
    for idx, action in idx_to_action.items():
        action_total = total_by_class[idx]
        action_predicted = predicted_by_class[idx] if predicted_by_class is not None else 0
        if action_total == 0:
            continue
        action_accuracy = correct_by_class[idx] / action_total
        action_precision = correct_by_class[idx] / max(1, action_predicted)
        action_recall = action_accuracy
        action_f1 = (
            2.0 * action_precision * action_recall / max(1e-8, action_precision + action_recall)
        )
        action_accuracies.append(action_accuracy)
        action_f1_scores.append(action_f1)
        per_action[action] = {
            "accuracy": action_accuracy,
            "precision": action_precision,
            "recall": action_recall,
            "f1": action_f1,
            "count": float(action_total),
            "predicted_count": float(action_predicted),
        }

    metrics = {
        "loss": loss_sum / max(1, total),
        "accuracy": correct / max(1, total),
        "balanced_accuracy": sum(action_accuracies) / max(1, len(action_accuracies)),
        "macro_f1": sum(action_f1_scores) / max(1, len(action_f1_scores)),
        "count": float(total),
        "per_action": per_action,
    }
    if topk_correct is not None:
        for k, value in sorted(topk_correct.items()):
            metrics[f"top{k}_accuracy"] = value / max(1, total)
    if score_metrics is not None and score_metrics.get("count", 0.0) > 0:
        score_count = max(1.0, score_metrics["count"])
        ratio_count = max(1.0, score_metrics.get("gain_ratio_count", 0.0))
        metrics["oracle_gain"] = {
            "mean_predicted_score": score_metrics["predicted_score_sum"] / score_count,
            "mean_best_score": score_metrics["best_score_sum"] / score_count,
            "positive_gain_rate": score_metrics["positive_gain_count"] / score_count,
            "nonnegative_gain_rate": score_metrics["nonnegative_gain_count"] / score_count,
            "negative_gain_rate": score_metrics["negative_gain_count"] / score_count,
            "best_positive_gain_rate": score_metrics["best_positive_gain_count"] / score_count,
            "mean_gain_ratio_when_best_positive": score_metrics["gain_ratio_sum"] / ratio_count,
        }
        if "relative_predicted_gain_sum" in score_metrics:
            metrics["oracle_gain"]["mean_predicted_relative_gain"] = (
                score_metrics["relative_predicted_gain_sum"] / score_count
            )
            metrics["oracle_gain"]["mean_best_relative_gain"] = (
                score_metrics["relative_best_gain_sum"] / score_count
            )
        metrics["balanced_gain_score"] = 0.5 * metrics["macro_f1"] + 0.5 * metrics["oracle_gain"]["positive_gain_rate"]
        metrics["safe_balanced_gain_score"] = (
            0.4 * metrics["macro_f1"]
            + 0.4 * metrics["oracle_gain"]["positive_gain_rate"]
            + 0.2 * metrics["oracle_gain"]["nonnegative_gain_rate"]
        )
    return metrics


def empty_oracle_score_metrics() -> Dict[str, float]:
    return {
        "count": 0.0,
        "predicted_score_sum": 0.0,
        "best_score_sum": 0.0,
        "positive_gain_count": 0.0,
        "nonnegative_gain_count": 0.0,
        "negative_gain_count": 0.0,
        "best_positive_gain_count": 0.0,
        "gain_ratio_sum": 0.0,
        "gain_ratio_count": 0.0,
        "relative_predicted_gain_sum": 0.0,
        "relative_best_gain_sum": 0.0,
    }


def build_class_weights(
    records: List[Dict[str, Any]],
    action_vocab: Dict[str, int],
    max_weight: Optional[float],
    device: str,
) -> Any:
    import torch

    counts = Counter(record["label"] for record in records)
    weights = np.ones(len(action_vocab), dtype=np.float32)
    nonzero_counts = [count for action, count in counts.items() if action in action_vocab and count > 0]
    total = float(sum(nonzero_counts))
    class_count = max(1, len(nonzero_counts))
    for action, idx in action_vocab.items():
        count = counts.get(action, 0)
        if count > 0:
            weights[idx] = total / (class_count * float(count))

    if max_weight is not None:
        weights = np.minimum(weights, float(max_weight))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def build_balanced_sampler(
    records: List[Dict[str, Any]],
    action_vocab: Dict[str, int],
) -> Any:
    import torch
    from torch.utils.data import WeightedRandomSampler

    counts = Counter(record["label"] for record in records)
    weights = [
        1.0 / max(1, counts.get(record["label"], 0))
        for record in records
        if record.get("label") in action_vocab
    ]
    return WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )


def supervised_classification_loss(
    logits: Any,
    labels: Any,
    class_weights: Any,
    focal_gamma: float,
) -> Any:
    import torch
    import torch.nn.functional as F

    ce = F.cross_entropy(logits, labels, weight=class_weights, reduction="none")
    if focal_gamma > 0:
        pt = torch.exp(-ce.detach()).clamp(1.0e-6, 1.0)
        ce = ((1.0 - pt) ** focal_gamma) * ce
    return ce.mean()


def oracle_soft_target_loss(
    logits: Any,
    batch: Dict[str, Any],
    temperature: float,
) -> Any:
    import torch
    import torch.nn.functional as F

    mask = batch["candidate_mask"] > 0
    valid = mask.any(dim=1)
    if not bool(valid.any()):
        return logits.new_tensor(0.0)

    scores = batch["candidate_scores"][valid]
    mask = mask[valid]
    scaled_scores = scores / max(temperature, 1e-6)
    scaled_scores = scaled_scores.masked_fill(~mask, -1.0e9)
    target_probs = torch.softmax(scaled_scores, dim=1) * mask.float()
    target_probs = target_probs / target_probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return F.kl_div(
        torch.log_softmax(logits[valid], dim=1),
        target_probs,
        reduction="batchmean",
    )


def gain_score_regression_loss(
    predictions: Any,
    batch: Dict[str, Any],
    score_scale: float,
) -> Any:
    import torch.nn.functional as F

    mask = batch["candidate_mask"] > 0
    if not bool(mask.any()):
        return predictions.new_tensor(0.0)
    target = batch["candidate_scores"] / max(score_scale, 1.0e-6)
    return F.smooth_l1_loss(predictions[mask], target[mask], reduction="mean")


def gain_ranking_loss(
    predictions: Any,
    batch: Dict[str, Any],
    score_scale: float,
    margin: float,
) -> Any:
    import torch
    import torch.nn.functional as F

    scores = batch["candidate_scores"]
    mask = batch["candidate_mask"] > 0
    valid = mask.any(dim=1)
    if not bool(valid.any()):
        return predictions.new_tensor(0.0)

    predictions = predictions[valid]
    scores = scores[valid]
    mask = mask[valid]
    masked_scores = scores.masked_fill(~mask, -1.0e9)
    best_idx = masked_scores.argmax(dim=1)
    best_pred = predictions.gather(1, best_idx.unsqueeze(1))
    target_gap = (masked_scores.gather(1, best_idx.unsqueeze(1)) - scores) / max(score_scale, 1.0e-6)
    effective_margin = torch.clamp(target_gap, min=0.0, max=margin)
    losses = F.relu(effective_margin - (best_pred - predictions))
    losses = losses.masked_fill(~mask, 0.0)
    losses.scatter_(1, best_idx.unsqueeze(1), 0.0)
    denom = (mask.sum(dim=1) - 1).clamp_min(1).to(losses.dtype)
    return (losses.sum(dim=1) / denom).mean()


def gain_score_loss(
    predictions: Any,
    batch: Dict[str, Any],
    score_scale: float,
    score_loss_weight: float,
    ranking_loss_weight: float,
    ranking_margin: float,
    ce_loss_weight: float,
    class_weights: Any,
    focal_gamma: float,
) -> tuple[Any, Dict[str, float]]:
    reg_loss = gain_score_regression_loss(predictions, batch, score_scale)
    rank_loss = gain_ranking_loss(predictions, batch, score_scale, ranking_margin)
    if ce_loss_weight > 0:
        ce_loss = supervised_classification_loss(
            predictions,
            batch["label"],
            class_weights,
            focal_gamma,
        )
    else:
        ce_loss = predictions.new_tensor(0.0)
    loss = (
        score_loss_weight * reg_loss
        + ranking_loss_weight * rank_loss
        + ce_loss_weight * ce_loss
    )
    return loss, {
        "score_regression_loss": float(reg_loss.detach().cpu().item()),
        "ranking_loss": float(rank_loss.detach().cpu().item()),
        "ce_loss": float(ce_loss.detach().cpu().item()),
    }


def select_predictions(
    logits: Any,
    batch: Dict[str, Any],
    idx_to_action: Dict[int, str],
    objective: str,
    score_scale: float,
    stay_threshold: Optional[float],
    stay_ratio_threshold: Optional[float],
    stay_ratio_epsilon: float,
) -> Any:
    import torch

    if objective != "gain_score" or (stay_threshold is None and stay_ratio_threshold is None):
        return logits.argmax(dim=1)

    stay_idx = None
    for idx, action in idx_to_action.items():
        if action == "Stay":
            stay_idx = int(idx)
            break
    if stay_idx is None:
        return logits.argmax(dim=1)

    moving_scores = logits.clone()
    moving_scores[:, stay_idx] = -torch.inf
    best_moving_scores, best_moving_idx = moving_scores.max(dim=1)
    if stay_ratio_threshold is not None and "initial_visible_pixels" in batch:
        predicted_gain = best_moving_scores * float(score_scale)
        base_pixels = batch["initial_visible_pixels"].to(predicted_gain.device).float()
        predicted_ratio = (base_pixels + predicted_gain) / base_pixels.clamp_min(float(stay_ratio_epsilon))
        choose_stay = predicted_ratio <= float(stay_ratio_threshold)
    else:
        scaled_threshold = float(stay_threshold) / max(score_scale, 1.0e-6)
        choose_stay = best_moving_scores <= scaled_threshold
    stay_pred = torch.full_like(best_moving_idx, stay_idx)
    return torch.where(choose_stay, stay_pred, best_moving_idx)


def evaluate(
    model: Any,
    loader: Any,
    device: str,
    idx_to_action: Optional[Dict[int, str]] = None,
    objective: str = "classification",
    score_scale: float = 1000.0,
    score_loss_weight: float = 1.0,
    ranking_loss_weight: float = 0.5,
    ranking_margin: float = 1.0,
    ce_loss_weight: float = 0.1,
    stay_threshold: Optional[float] = None,
    stay_ratio_threshold: Optional[float] = None,
    stay_ratio_epsilon: float = 1.0,
) -> Dict[str, Any]:
    import torch

    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    if idx_to_action is None:
        idx_to_action = {}
    num_actions = len(idx_to_action)
    correct_by_class = [0 for _ in range(num_actions)]
    total_by_class = [0 for _ in range(num_actions)]
    predicted_by_class = [0 for _ in range(num_actions)]
    topk_correct = {2: 0, 3: 0}
    score_metrics = empty_oracle_score_metrics()
    score_loss_sum = 0.0
    ranking_loss_sum = 0.0
    ce_loss_sum = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            logits = model(batch)
            if not idx_to_action:
                idx_to_action = {idx: str(idx) for idx in range(logits.shape[1])}
                num_actions = len(idx_to_action)
                correct_by_class = [0 for _ in range(num_actions)]
                total_by_class = [0 for _ in range(num_actions)]
                predicted_by_class = [0 for _ in range(num_actions)]
            if objective == "gain_score":
                loss, loss_parts = gain_score_loss(
                    logits,
                    batch,
                    score_scale=score_scale,
                    score_loss_weight=score_loss_weight,
                    ranking_loss_weight=ranking_loss_weight,
                    ranking_margin=ranking_margin,
                    ce_loss_weight=ce_loss_weight,
                    class_weights=None,
                    focal_gamma=0.0,
                )
                batch_count = int(batch["label"].numel())
                score_loss_sum += loss_parts["score_regression_loss"] * batch_count
                ranking_loss_sum += loss_parts["ranking_loss"] * batch_count
                ce_loss_sum += loss_parts["ce_loss"] * batch_count
            else:
                loss = criterion(logits, batch["label"])
            loss_sum += float(loss.item()) * batch["label"].numel()
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
            correct += int((pred == batch["label"]).sum().item())
            total += int(batch["label"].numel())
            update_class_counts(pred, batch["label"], correct_by_class, total_by_class, predicted_by_class)
            update_topk_counts(logits, batch["label"], topk_correct)
            update_oracle_score_metrics(pred, batch, score_metrics)
    metrics = classification_metrics(
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
    if objective == "gain_score":
        metrics["objective"] = objective
        metrics["score_regression_loss"] = score_loss_sum / max(1, total)
        metrics["ranking_loss"] = ranking_loss_sum / max(1, total)
        metrics["ce_loss"] = ce_loss_sum / max(1, total)
    if stay_threshold is not None:
        metrics["selection"] = {"stay_threshold": float(stay_threshold)}
    if stay_ratio_threshold is not None:
        metrics["selection"] = {
            "stay_ratio_threshold": float(stay_ratio_threshold),
            "stay_ratio_epsilon": float(stay_ratio_epsilon),
        }
    return metrics


def train(config: Dict[str, Any]) -> None:
    import torch
    from torch.utils.data import DataLoader

    seed = int(config.get("seed", 7))
    set_seed(seed)

    data_cfg = config["data"]
    train_cfg = config["train"]
    data_dir = Path(data_cfg["data_dir"])
    records = read_jsonl(Path(data_cfg["records_path"]))
    if not records:
        raise ValueError("No records found. Generate AI2-THOR data first.")

    split_strategy = str(data_cfg.get("split_strategy", "random"))
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
        train_records, val_records = split_records(
            records,
            val_fraction=val_fraction,
            seed=seed,
        )
        split_info = {
            "strategy": split_strategy,
            "train_count": len(train_records),
            "val_count": len(val_records),
        }
    else:
        raise ValueError(f"Unsupported split strategy: {split_strategy}")

    if not train_records:
        train_records = list(records)
        val_records = []
        split_info["train_count"] = len(train_records)
        split_info["val_count"] = 0

    action_vocab = build_action_vocab(records, config.get("candidate_actions"))
    idx_to_action = {idx: action for action, idx in action_vocab.items()}
    text_vocab = build_text_vocab(train_records, max_size=int(data_cfg.get("max_vocab_size", 4096)))
    dataset_kwargs = {
        "data_dir": data_dir,
        "text_vocab": text_vocab,
        "action_vocab": action_vocab,
        "image_size": int(data_cfg.get("image_size", 128)),
        "max_depth": float(data_cfg.get("max_depth", 5.0)),
        "max_text_len": int(data_cfg.get("max_text_len", 32)),
        "ablation": data_cfg.get("ablation", config.get("ablation", {})),
    }
    train_dataset = AI2ThorOracleDataset(train_records, **dataset_kwargs)
    val_dataset = AI2ThorOracleDataset(val_records, **dataset_kwargs) if val_records else None

    device = str(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = build_model(config, len(text_vocab), len(action_vocab)).to(device)
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
            T_max=int(train_cfg.get("epochs", 5)),
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
        batch_size=int(train_cfg.get("batch_size", 16)),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(train_cfg.get("num_workers", 0)),
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=int(train_cfg.get("batch_size", 16)),
            shuffle=False,
            num_workers=int(train_cfg.get("num_workers", 0)),
        )
        if val_dataset is not None
        else None
    )

    output_dir = Path(train_cfg.get("output_dir", "runs/ai2thor_policy"))
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
    if soft_target_weight > 0:
        print(
            json.dumps(
                {
                    "loss": {
                        "soft_target_weight": soft_target_weight,
                        "soft_target_temperature": soft_target_temperature,
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
    for epoch in range(1, int(train_cfg.get("epochs", 5)) + 1):
        model.train()
        total = 0
        correct = 0
        loss_sum = 0.0
        ce_loss_sum = 0.0
        soft_loss_sum = 0.0
        correct_by_class = [0 for _ in range(len(action_vocab))]
        total_by_class = [0 for _ in range(len(action_vocab))]
        predicted_by_class = [0 for _ in range(len(action_vocab))]
        topk_correct = {2: 0, 3: 0}
        score_metrics = empty_oracle_score_metrics()
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
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
            loss.backward()
            if gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()

            batch_count = int(batch["label"].numel())
            loss_sum += float(loss.item()) * batch_count
            ce_loss_sum += float(ce_loss.item()) * batch_count
            soft_loss_sum += float(soft_loss.item()) * batch_count
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
        if soft_target_weight > 0:
            train_metrics["soft_target_loss"] = soft_loss_sum / max(1, total)
        val_metrics = evaluate(model, val_loader, device, idx_to_action) if val_loader is not None else {}
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
                    "text_vocab": text_vocab,
                    "action_vocab": action_vocab,
                    "class_weights": class_weights.detach().cpu().tolist() if class_weights is not None else None,
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
    parser.add_argument("--config", type=Path, default=Path("configs/train_ai2thor_policy.yaml"))
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
