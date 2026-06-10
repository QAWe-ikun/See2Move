from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml
from PIL import Image


@dataclass
class CandidateResult:
    action: str
    success: bool
    visible_pixels: int
    score: float


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_instruction(object_type: str) -> str:
    return f"Move the camera to get a clearer view of the {object_type.lower()}."


def visible_pixels(instance_masks: Dict[str, Any], object_id: str) -> int:
    mask = instance_masks.get(object_id)
    if mask is None:
        return 0
    return int(np.asarray(mask, dtype=np.uint8).sum())


def choose_target(objects: Iterable[Dict[str, Any]], allowed_types: List[str]) -> Optional[Dict[str, Any]]:
    candidates = [
        obj
        for obj in objects
        if obj.get("visible") and obj.get("objectType") in allowed_types
    ]
    if not candidates:
        return None
    return random.choice(candidates)


def save_observation(event: Any, output_dir: Path, sample_id: int) -> Tuple[str, str]:
    rgb_path = output_dir / f"{sample_id:06d}_rgb.png"
    depth_path = output_dir / f"{sample_id:06d}_depth.npy"

    Image.fromarray(event.frame).save(rgb_path)
    np.save(depth_path, event.depth_frame)
    return rgb_path.name, depth_path.name


def apply_action(controller: Any, action: str, cfg: Dict[str, Any]) -> Any:
    if action in {"RotateLeft", "RotateRight"}:
        return controller.step(action=action, degrees=cfg["rotate_step_degrees"])
    if action in {"LookUp", "LookDown"}:
        return controller.step(action=action, degrees=cfg["horizon_step_degrees"])
    return controller.step(action=action)


def evaluate_candidates(controller: Any, actions: List[str], target_id: str, cfg: Dict[str, Any]) -> List[CandidateResult]:
    base_event = controller.last_event
    base_pose = dict(base_event.metadata["agent"])
    base_pixels = visible_pixels(base_event.instance_masks, target_id)

    results: List[CandidateResult] = []
    for action in actions:
        event = apply_action(controller, action, cfg)
        success = bool(event.metadata["lastActionSuccess"])
        pixels = visible_pixels(event.instance_masks, target_id)
        score = float(pixels - base_pixels if success else -1)

        controller.step(
            action="TeleportFull",
            x=base_pose["position"]["x"],
            y=base_pose["position"]["y"],
            z=base_pose["position"]["z"],
            rotation=base_pose["rotation"],
            horizon=base_pose["cameraHorizon"],
            standing=base_pose.get("isStanding", True),
        )
        results.append(CandidateResult(action, success, pixels, score))
    return results


def generate_dataset(cfg: Dict[str, Any]) -> None:
    from ai2thor.controller import Controller

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    controller = Controller(
        scene=cfg["scene"],
        width=cfg["width"],
        height=cfg["height"],
        gridSize=cfg["grid_size"],
        renderDepthImage=True,
        renderInstanceSegmentation=True,
    )

    records = []
    try:
        for sample_id in range(int(cfg["max_episodes"])):
            event = controller.reset(scene=cfg["scene"])
            reachable = controller.step(action="GetReachablePositions").metadata["actionReturn"]
            pose = random.choice(reachable)
            controller.step(
                action="TeleportFull",
                x=pose["x"],
                y=pose["y"],
                z=pose["z"],
                rotation={"x": 0, "y": random.choice([0, 90, 180, 270]), "z": 0},
                horizon=random.choice([0, 15, 30]),
                standing=True,
            )
            event = controller.last_event
            target = choose_target(event.metadata["objects"], cfg["target_object_types"])
            if target is None:
                continue

            rgb_file, depth_file = save_observation(event, output_dir, sample_id)
            candidates = evaluate_candidates(
                controller,
                cfg["candidate_actions"],
                target["objectId"],
                cfg,
            )
            best = max(candidates, key=lambda c: c.score)
            initial_visible_pixels = visible_pixels(event.instance_masks, target["objectId"])
            records.append(
                {
                    "id": sample_id,
                    "scene": cfg["scene"],
                    "instruction": make_instruction(target["objectType"]),
                    "target": {
                        "object_id": target["objectId"],
                        "object_type": target["objectType"],
                    },
                    "initial_visible_pixels": initial_visible_pixels,
                    "rgb": rgb_file,
                    "depth": depth_file,
                    "agent": event.metadata["agent"],
                    "candidates": [c.__dict__ for c in candidates],
                    "label": best.action,
                }
            )
    finally:
        controller.stop()

    with (output_dir / "records.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/ai2thor_oracle.yaml"))
    args = parser.parse_args()
    generate_dataset(load_config(args.config))


if __name__ == "__main__":
    main()
