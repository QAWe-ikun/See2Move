from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from see2move.data.ai2thor_records import (
    AI2ThorOracleDataset,
    build_action_vocab,
    build_text_vocab,
    read_jsonl,
    split_records,
)
from see2move.models.policy import build_model


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    return {key: value.to(device) for key, value in batch.items()}


def evaluate(model: Any, loader: Any, device: str) -> Dict[str, float]:
    import torch

    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            logits = model(batch)
            loss = criterion(logits, batch["label"])
            loss_sum += float(loss.item()) * batch["label"].numel()
            pred = logits.argmax(dim=1)
            correct += int((pred == batch["label"]).sum().item())
            total += int(batch["label"].numel())
    return {
        "loss": loss_sum / max(1, total),
        "accuracy": correct / max(1, total),
        "count": float(total),
    }


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

    action_vocab = build_action_vocab(records, config.get("candidate_actions"))
    text_vocab = build_text_vocab(records, max_size=int(data_cfg.get("max_vocab_size", 4096)))
    train_records, val_records = split_records(
        records,
        val_fraction=float(data_cfg.get("val_fraction", 0.2)),
        seed=seed,
    )
    if not train_records:
        train_records = list(records)
        val_records = []

    dataset_kwargs = {
        "data_dir": data_dir,
        "text_vocab": text_vocab,
        "action_vocab": action_vocab,
        "image_size": int(data_cfg.get("image_size", 128)),
        "max_depth": float(data_cfg.get("max_depth", 5.0)),
        "max_text_len": int(data_cfg.get("max_text_len", 32)),
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
    criterion = torch.nn.CrossEntropyLoss()

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg.get("batch_size", 16)),
        shuffle=True,
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

    metrics = []
    best_accuracy = -1.0
    best_path = output_dir / "checkpoint_best.pt"
    for epoch in range(1, int(train_cfg.get("epochs", 5)) + 1):
        model.train()
        total = 0
        correct = 0
        loss_sum = 0.0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch)
            loss = criterion(logits, batch["label"])
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.item()) * batch["label"].numel()
            pred = logits.argmax(dim=1)
            correct += int((pred == batch["label"]).sum().item())
            total += int(batch["label"].numel())

        train_metrics = {
            "loss": loss_sum / max(1, total),
            "accuracy": correct / max(1, total),
            "count": float(total),
        }
        val_metrics = evaluate(model, val_loader, device) if val_loader is not None else {}
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        metrics.append(row)
        print(json.dumps(row))

        score = float(val_metrics.get("accuracy", train_metrics["accuracy"]))
        if score >= best_accuracy:
            best_accuracy = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "text_vocab": text_vocab,
                    "action_vocab": action_vocab,
                    "epoch": epoch,
                    "metrics": row,
                },
                best_path,
            )

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
