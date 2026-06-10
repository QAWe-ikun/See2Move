from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image


COMMANDS: Dict[str, str] = {
    "w": "MoveAhead",
    "s": "MoveBack",
    "a": "MoveLeft",
    "d": "MoveRight",
    "j": "RotateLeft",
    "l": "RotateRight",
    "i": "LookUp",
    "k": "LookDown",
}


def save_frames(event: Any, output_dir: Path, prefix: str = "latest") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(event.frame).save(output_dir / f"{prefix}_rgb.png")

    if event.depth_frame is not None:
        depth = np.nan_to_num(event.depth_frame, nan=0.0, posinf=0.0, neginf=0.0)
        max_value = float(depth.max()) if depth.size else 0.0
        if max_value > 0:
            depth_vis = np.clip(depth / max_value, 0.0, 1.0)
        else:
            depth_vis = depth
        Image.fromarray((depth_vis * 255).astype(np.uint8)).save(
            output_dir / f"{prefix}_depth.png"
        )
        np.save(output_dir / f"{prefix}_depth.npy", event.depth_frame)


def visible_object_summary(event: Any, limit: int = 15) -> str:
    visible = [
        obj["objectType"]
        for obj in event.metadata["objects"]
        if obj.get("visible")
    ]
    if not visible:
        return "visible objects: none"
    return "visible objects: " + ", ".join(visible[:limit])


def print_status(event: Any, output_dir: Path) -> None:
    agent = event.metadata["agent"]
    pos = agent["position"]
    rot = agent["rotation"]
    print(
        "pose: "
        f"x={pos['x']:.2f}, y={pos['y']:.2f}, z={pos['z']:.2f}, "
        f"yaw={rot['y']:.1f}, horizon={agent['cameraHorizon']:.1f}"
    )
    print(visible_object_summary(event))
    print(f"saved: {output_dir / 'latest_rgb.png'}")


def random_teleport(controller: Any) -> Any:
    import random

    reachable = controller.step(action="GetReachablePositions").metadata["actionReturn"]
    pose = random.choice(reachable)
    return controller.step(
        action="TeleportFull",
        x=pose["x"],
        y=pose["y"],
        z=pose["z"],
        rotation={"x": 0, "y": random.choice([0, 90, 180, 270]), "z": 0},
        horizon=random.choice([0, 15, 30]),
        standing=True,
    )


def apply_action(controller: Any, command: str, rotate_step: int, horizon_step: int) -> Any:
    action = COMMANDS[command]
    if action in {"RotateLeft", "RotateRight"}:
        return controller.step(action=action, degrees=rotate_step)
    if action in {"LookUp", "LookDown"}:
        return controller.step(action=action, degrees=horizon_step)
    return controller.step(action=action)


def main() -> None:
    from ai2thor.controller import Controller

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="FloorPlan1")
    parser.add_argument("--width", type=int, default=600)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--grid-size", type=float, default=0.25)
    parser.add_argument("--rotate-step", type=int, default=30)
    parser.add_argument("--horizon-step", type=int, default=15)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ai2thor_explorer"))
    args = parser.parse_args()

    controller = Controller(
        scene=args.scene,
        width=args.width,
        height=args.height,
        gridSize=args.grid_size,
        renderDepthImage=True,
        renderInstanceSegmentation=True,
    )

    try:
        event = controller.last_event
        save_frames(event, args.output_dir)
        print("Controls: w/s/a/d move, j/l rotate, i/k look, r random teleport, o objects, q quit")
        print_status(event, args.output_dir)
        while True:
            command = input("> ").strip().lower()
            if command in {"q", "quit", "exit"}:
                break
            if command == "r":
                event = random_teleport(controller)
            elif command == "o":
                print(visible_object_summary(controller.last_event, limit=50))
                continue
            elif command in COMMANDS:
                event = apply_action(controller, command, args.rotate_step, args.horizon_step)
                if not event.metadata["lastActionSuccess"]:
                    print(f"action failed: {event.metadata.get('errorMessage', '')}")
            else:
                print("unknown command")
                continue
            save_frames(event, args.output_dir)
            print_status(event, args.output_dir)
    finally:
        controller.stop()


if __name__ == "__main__":
    main()
