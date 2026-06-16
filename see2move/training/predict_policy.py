from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from see2move.data.ai2thor_records import AI2ThorOracleDataset
from see2move.models.policy import build_model


def main() -> None:
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("/mnt/f/see2move/data/ai2thor_oracle_10k_stay"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt: Dict[str, Any] = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt["config"]
    data_cfg = config["data"]
    record = json.loads(args.record.read_text(encoding="utf-8"))
    dataset = AI2ThorOracleDataset(
        [record],
        data_dir=args.data_dir,
        text_vocab=ckpt["text_vocab"],
        action_vocab=ckpt["action_vocab"],
        image_size=int(data_cfg.get("image_size", 128)),
        max_depth=float(data_cfg.get("max_depth", 5.0)),
        max_text_len=int(data_cfg.get("max_text_len", 32)),
        ablation=data_cfg.get("ablation", config.get("ablation", {})),
    )
    batch = dataset[0]
    batch = {key: value.unsqueeze(0).to(args.device) for key, value in batch.items() if key != "label"}

    model = build_model(config, len(ckpt["text_vocab"]), len(ckpt["action_vocab"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(args.device)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(batch), dim=1).squeeze(0).cpu()

    idx_to_action = {idx: action for action, idx in ckpt["action_vocab"].items()}
    best_idx = int(probs.argmax().item())
    result = {
        "prediction": idx_to_action[best_idx],
        "confidence": float(probs[best_idx].item()),
        "scores": {idx_to_action[i]: float(probs[i].item()) for i in range(len(idx_to_action))},
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
