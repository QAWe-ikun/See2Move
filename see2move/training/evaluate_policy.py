from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from see2move.data.ai2thor_records import AI2ThorOracleDataset, read_jsonl
from see2move.models.policy import build_model
from see2move.training.train_policy import evaluate


def main() -> None:
    import torch
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--records", type=Path, default=Path("data/ai2thor_oracle/records.jsonl"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/ai2thor_oracle"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    ckpt: Dict[str, Any] = torch.load(args.checkpoint, map_location="cpu")
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
    )
    loader = DataLoader(dataset, batch_size=int(config["train"].get("batch_size", 16)))
    model = build_model(config, len(ckpt["text_vocab"]), len(ckpt["action_vocab"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(args.device)
    print(json.dumps(evaluate(model, loader, args.device), indent=2))


if __name__ == "__main__":
    main()
