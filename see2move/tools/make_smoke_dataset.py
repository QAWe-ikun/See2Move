from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def make_record(sample_id: int, action: str) -> dict:
    return {
        "id": sample_id,
        "scene": "SmokeScene",
        "instruction": f"Move the camera to get a clearer view of the chair {sample_id}.",
        "target": {"object_id": f"Chair|{sample_id}", "object_type": "Chair"},
        "initial_visible_pixels": int(100 + sample_id),
        "history": [],
        "rgb": f"{sample_id:06d}_rgb.png",
        "depth": f"{sample_id:06d}_depth.npy",
        "agent": {
            "position": {"x": float(sample_id % 3), "y": 0.9, "z": float(sample_id % 5)},
            "rotation": {"x": 0.0, "y": float((sample_id % 4) * 90), "z": 0.0},
            "cameraHorizon": float((sample_id % 3) * 15),
            "isStanding": True,
        },
        "candidates": [
            {"action": "MoveAhead", "success": True, "visible_pixels": 120, "score": 20.0},
            {"action": "RotateLeft", "success": True, "visible_pixels": 90, "score": -10.0},
            {"action": "RotateRight", "success": True, "visible_pixels": 80, "score": -20.0},
            {"action": "Stay", "success": True, "visible_pixels": 100 + sample_id, "score": 0.0},
        ],
        "label": action,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data/smoke_ai2thor"))
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=64)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    actions = ["MoveAhead", "RotateLeft", "RotateRight", "Stay"]
    rng = np.random.default_rng(7)
    records = []
    for sample_id in range(args.count):
        action = actions[sample_id % len(actions)]
        rgb = rng.integers(0, 255, size=(args.image_size, args.image_size, 3), dtype=np.uint8)
        depth = rng.random((args.image_size, args.image_size), dtype=np.float32) * 5.0
        Image.fromarray(rgb).save(args.output_dir / f"{sample_id:06d}_rgb.png")
        np.save(args.output_dir / f"{sample_id:06d}_depth.npy", depth)
        records.append(make_record(sample_id, action))

    with (args.output_dir / "records.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
