from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from see2move.data.ai2thor_records import build_action_vocab, read_jsonl
from see2move.models.policy import build_qwen_model
from see2move.training.train_policy import evaluate
from see2move.training.train_qwen_policy import QwenFeatureOracleDataset, load_qwen_feature_payload


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--records", type=Path, default=Path("/mnt/f/see2move/data/ai2thor_oracle_10k_stay/records.jsonl"))
    parser.add_argument("--data-dir", type=Path, default=Path("/mnt/f/see2move/data/ai2thor_oracle_10k_stay"))
    parser.add_argument("--qwen-features", type=Path, default=Path("/mnt/f/see2move/features/qwen3vl_ai2thor_oracle_10k_stay.pt"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt: Dict[str, Any] = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt["config"]
    data_cfg = config["data"]
    records = read_jsonl(args.records)
    action_vocab = ckpt.get("action_vocab") or build_action_vocab(records, config.get("candidate_actions"))
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
    loader = DataLoader(dataset, batch_size=int(config["train"].get("batch_size", 64)))
    model = build_qwen_model(
        config,
        qwen_dim=int(ckpt.get("qwen_feature_dim", feature_payload["features"].shape[1])),
        num_actions=len(action_vocab),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(args.device)
    idx_to_action = {idx: action for action, idx in action_vocab.items()}
    print(json.dumps(evaluate(model, loader, args.device, idx_to_action), indent=2))


if __name__ == "__main__":
    main()
