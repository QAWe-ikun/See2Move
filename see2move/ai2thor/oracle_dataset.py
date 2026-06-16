from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class CandidateResult:
    action: str
    success: bool
    visible_pixels: int
    score: float


STAY_ACTIONS = {"Stay", "NoOp", "Stop"}


def is_stay_action(action: str) -> bool:
    return action in STAY_ACTIONS


def choose_best_candidate(candidates: List[CandidateResult]) -> CandidateResult:
    stay = next((candidate for candidate in candidates if is_stay_action(candidate.action)), None)
    moving_candidates = [
        candidate for candidate in candidates if not is_stay_action(candidate.action)
    ]
    if stay is not None:
        best_moving_score = max(
            (candidate.score for candidate in moving_candidates),
            default=float("-inf"),
        )
        if best_moving_score <= 0.0:
            return stay
    return max(candidates, key=lambda candidate: candidate.score)


def load_config(path: Path) -> Dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def expand_scene_entry(entry: str) -> List[str]:
    match = re.fullmatch(r"([A-Za-z_]+)(\d+)-(\d+)", entry)
    if match is None:
        return [entry]

    prefix, start_text, end_text = match.groups()
    start = int(start_text)
    end = int(end_text)
    step = 1 if end >= start else -1
    return [f"{prefix}{idx}" for idx in range(start, end + step, step)]


def configured_scenes(cfg: Dict[str, Any]) -> List[str]:
    entries = cfg["scenes"]
    if isinstance(entries, str):
        entries = [entries]

    scenes: List[str] = []
    for entry in entries:
        scenes.extend(expand_scene_entry(str(entry)))
    if not scenes:
        raise ValueError("At least one AI2-THOR scene must be configured.")
    return scenes


def humanize_object_type(object_type: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", " ", object_type).lower()


def make_instruction(object_type: str) -> str:
    return f"Move the camera to get a clearer view of the {humanize_object_type(object_type)}."


def visible_pixels(instance_masks: Dict[str, Any], object_id: str) -> int:
    import numpy as np

    mask = instance_masks.get(object_id)
    if mask is None:
        return 0
    return int(np.asarray(mask, dtype=np.uint8).sum())


def read_existing_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def next_record_id(records: List[Dict[str, Any]]) -> int:
    if not records:
        return 0
    return max(int(record.get("id", idx)) for idx, record in enumerate(records)) + 1


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
    import numpy as np
    from PIL import Image

    rgb_path = output_dir / f"{sample_id:06d}_rgb.png"
    depth_path = output_dir / f"{sample_id:06d}_depth.npy"

    Image.fromarray(event.frame).save(rgb_path)
    np.save(depth_path, event.depth_frame)
    return rgb_path.name, depth_path.name


def apply_action(controller: Any, action: str, cfg: Dict[str, Any]) -> Any:
    if is_stay_action(action):
        return controller.last_event
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
        if is_stay_action(action):
            results.append(CandidateResult(action, True, base_pixels, 0.0))
            continue

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
    import numpy as np
    from ai2thor.controller import Controller
    from tqdm.auto import tqdm

    seed = cfg.get("seed")
    if seed is not None:
        random.seed(int(seed))
        np.random.seed(int(seed))

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    resume = bool(cfg.get("resume", True))
    records = read_existing_records(records_path) if resume else []
    if not resume:
        records_path.write_text("", encoding="utf-8")

    scenes = configured_scenes(cfg)
    scene_order = list(scenes)
    random.shuffle(scene_order)
    max_attempts = int(cfg.get("max_attempts", cfg.get("max_episodes", 0)))
    target_records = cfg.get("target_records")
    target_records = int(target_records) if target_records is not None else None
    if max_attempts <= 0:
        raise ValueError("Configure max_attempts or max_episodes with a positive value.")

    default_samples_per_scene = (
        1
        if target_records is not None
        else max(1, (max_attempts + len(scene_order) - 1) // len(scene_order))
    )
    samples_per_scene = int(cfg.get("samples_per_scene", default_samples_per_scene))

    controller = Controller(
        scene=scene_order[0],
        width=cfg["width"],
        height=cfg["height"],
        gridSize=cfg["grid_size"],
        renderDepthImage=True,
        renderInstanceSegmentation=True,
    )

    attempts = 0
    sample_id = next_record_id(records)
    progress_total = target_records if target_records is not None else max_attempts
    progress_initial = min(len(records), progress_total) if target_records is not None else 0
    progress = tqdm(
        total=progress_total,
        initial=progress_initial,
        desc="Generating AI2-THOR oracle",
        unit="record" if target_records is not None else "pose",
    )
    try:
        with records_path.open("a", encoding="utf-8") as f:
            while attempts < max_attempts and (
                target_records is None or len(records) < target_records
            ):
                for scene in scene_order:
                    if attempts >= max_attempts or (
                        target_records is not None and len(records) >= target_records
                    ):
                        break

                    controller.reset(scene=scene)
                    reachable = controller.step(action="GetReachablePositions").metadata["actionReturn"]
                    if not reachable:
                        continue

                    scene_attempts = min(samples_per_scene, max_attempts - attempts)
                    for _ in range(scene_attempts):
                        if target_records is not None and len(records) >= target_records:
                            break

                        attempts += 1
                        if target_records is None:
                            progress.update(1)
                        progress.set_postfix(scene=scene, attempts=attempts, records=len(records))

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
                        best = choose_best_candidate(candidates)
                        initial_visible_pixels = visible_pixels(event.instance_masks, target["objectId"])
                        record = {
                            "id": sample_id,
                            "scene": scene,
                            "instruction": make_instruction(target["objectType"]),
                            "target": {
                                "object_id": target["objectId"],
                                "object_type": target["objectType"],
                            },
                            "initial_visible_pixels": initial_visible_pixels,
                            "history": [],
                            "rgb": rgb_file,
                            "depth": depth_file,
                            "agent": event.metadata["agent"],
                            "candidates": [c.__dict__ for c in candidates],
                            "label": best.action,
                        }
                        records.append(record)
                        f.write(json.dumps(record) + "\n")
                        f.flush()
                        sample_id += 1
                        if target_records is not None:
                            progress.update(1)
                        progress.set_postfix(scene=scene, attempts=attempts, records=len(records))
    finally:
        progress.close()
        controller.stop()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/ai2thor_oracle.yaml"))
    args = parser.parse_args()
    generate_dataset(load_config(args.config))


if __name__ == "__main__":
    main()
