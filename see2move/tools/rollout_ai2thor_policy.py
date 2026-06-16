from __future__ import annotations

import argparse
import json
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List


def save_temp_observation(event: Any, temp_dir: Path, step_id: int) -> tuple[str, str]:
    import numpy as np
    from PIL import Image

    rgb = temp_dir / f"{step_id:06d}_rgb.png"
    depth = temp_dir / f"{step_id:06d}_depth.npy"
    Image.fromarray(event.frame).save(rgb)
    np.save(depth, event.depth_frame)
    return rgb.name, depth.name


def policy_action(
    model: Any,
    ckpt: Dict[str, Any],
    record: Dict[str, Any],
    temp_dir: Path,
    device: str,
) -> str:
    import torch

    from see2move.data.ai2thor_records import AI2ThorOracleDataset

    config = ckpt["config"]
    data_cfg = config["data"]
    dataset = AI2ThorOracleDataset(
        [record],
        data_dir=temp_dir,
        text_vocab=ckpt["text_vocab"],
        action_vocab=ckpt["action_vocab"],
        image_size=int(data_cfg.get("image_size", 128)),
        max_depth=float(data_cfg.get("max_depth", 5.0)),
        max_text_len=int(data_cfg.get("max_text_len", 32)),
        ablation=data_cfg.get("ablation", config.get("ablation", {})),
    )
    batch = {
        key: value.unsqueeze(0).to(device)
        for key, value in dataset[0].items()
        if key != "label"
    }
    with torch.no_grad():
        pred_idx = int(model(batch).argmax(dim=1).item())
    idx_to_action = {idx: action for action, idx in ckpt["action_vocab"].items()}
    return idx_to_action[pred_idx]


def rollout_episode(
    controller: Any,
    model: Any,
    ckpt: Dict[str, Any],
    cfg: Dict[str, Any],
    scene: str,
    temp_dir: Path,
    episode_id: int,
    steps: int,
    device: str,
    policy: str,
) -> Dict[str, Any] | None:
    from see2move.ai2thor.oracle_dataset import apply_action, choose_best_candidate, choose_target, evaluate_candidates, make_instruction, visible_pixels

    controller.reset(scene=scene)
    reachable = controller.step(action="GetReachablePositions").metadata["actionReturn"]
    if not reachable:
        return None

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
        return None

    target_id = target["objectId"]
    initial_pixels = visible_pixels(event.instance_masks, target_id)
    trace: List[Dict[str, Any]] = []
    for step_idx in range(steps):
        event = controller.last_event
        before = visible_pixels(event.instance_masks, target_id)
        candidates = evaluate_candidates(controller, cfg["candidate_actions"], target_id, cfg)
        best = choose_best_candidate(candidates)
        if policy == "oracle":
            action = best.action
        elif policy == "random":
            action = random.choice(cfg["candidate_actions"])
        else:
            rgb_file, depth_file = save_temp_observation(event, temp_dir, episode_id * 100 + step_idx)
            record = {
                "id": episode_id * 100 + step_idx,
                "scene": scene,
                "instruction": make_instruction(target["objectType"]),
                "target": {
                    "object_id": target_id,
                    "object_type": target["objectType"],
                },
                "initial_visible_pixels": before,
                "history": [],
                "rgb": rgb_file,
                "depth": depth_file,
                "agent": event.metadata["agent"],
                "candidates": [candidate.__dict__ for candidate in candidates],
                "label": best.action,
            }
            if record["label"] not in ckpt["action_vocab"]:
                record["label"] = next(iter(ckpt["action_vocab"]))
            action = policy_action(model, ckpt, record, temp_dir, device)

        event = apply_action(controller, action, cfg)
        success = True if action in {"Stay", "NoOp", "Stop"} else bool(event.metadata["lastActionSuccess"])
        after = visible_pixels(event.instance_masks, target_id)
        trace.append(
            {
                "step": step_idx + 1,
                "action": action,
                "oracle_action": best.action,
                "before_pixels": before,
                "after_pixels": after,
                "gain": after - before if success else -1,
                "success": success,
            }
        )

    final_pixels = visible_pixels(controller.last_event.instance_masks, target_id)
    return {
        "scene": scene,
        "target_type": target["objectType"],
        "initial_pixels": initial_pixels,
        "final_pixels": final_pixels,
        "total_gain": final_pixels - initial_pixels,
        "positive_total_gain": final_pixels > initial_pixels,
        "trace": trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/ai2thor_policy_10k_stay_simple_e35/checkpoint_best.pt"))
    parser.add_argument("--oracle-config", type=Path, default=Path("configs/ai2thor_oracle.yaml"))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--policy", choices=["model", "random", "oracle"], default="model")
    parser.add_argument("--output", type=Path, default=Path("runs/rollouts/simple_3step.json"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import numpy as np
    import torch
    from ai2thor.controller import Controller
    from tqdm.auto import tqdm
    from see2move.ai2thor.oracle_dataset import configured_scenes, load_config
    from see2move.models.policy import build_model

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = load_config(args.oracle_config)
    scenes = configured_scenes(cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(ckpt["config"], len(ckpt["text_vocab"]), len(ckpt["action_vocab"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(args.device)
    model.eval()

    controller = Controller(
        scene=scenes[0],
        width=cfg["width"],
        height=cfg["height"],
        gridSize=cfg["grid_size"],
        renderDepthImage=True,
        renderInstanceSegmentation=True,
    )
    results = []
    try:
        with tempfile.TemporaryDirectory() as tmp:
            temp_dir = Path(tmp)
            progress = tqdm(total=args.episodes, desc=f"{args.policy} rollout")
            attempts = 0
            while len(results) < args.episodes and attempts < args.episodes * 10:
                attempts += 1
                scene = random.choice(scenes)
                result = rollout_episode(
                    controller,
                    model,
                    ckpt,
                    cfg,
                    scene,
                    temp_dir,
                    len(results),
                    args.steps,
                    args.device,
                    args.policy,
                )
                if result is None:
                    continue
                results.append(result)
                progress.update(1)
            progress.close()
    finally:
        controller.stop()

    summary = {
        "policy": args.policy,
        "episodes": len(results),
        "steps": args.steps,
        "mean_total_gain": float(np.mean([item["total_gain"] for item in results])) if results else 0.0,
        "positive_total_gain_rate": float(np.mean([item["positive_total_gain"] for item in results])) if results else 0.0,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
