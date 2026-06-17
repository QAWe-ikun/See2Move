from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from see2move.ai2thor.oracle_dataset import apply_action, load_config, visible_pixels


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_font(size: int) -> ImageFont.ImageFont:
    for path in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def depth_preview(depth: np.ndarray) -> Image.Image:
    depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    high = float(np.percentile(depth, 95)) if depth.size else 1.0
    high = max(high, 1.0e-6)
    vis = np.clip(depth / high, 0.0, 1.0)
    return Image.fromarray((vis * 255).astype(np.uint8)).convert("RGB")


def save_event(event: Any, output_dir: Path, prefix: str) -> Dict[str, str]:
    rgb = Image.fromarray(event.frame).convert("RGB")
    depth = depth_preview(event.depth_frame)
    rgb_path = output_dir / f"{prefix}_rgb.png"
    depth_path = output_dir / f"{prefix}_depth.png"
    rgb.save(rgb_path)
    depth.save(depth_path)
    return {"rgb": str(rgb_path), "depth": str(depth_path)}


def teleport_to_record(controller: Any, record: Dict[str, Any]) -> Any:
    agent = record["agent"]
    pos = agent["position"]
    return controller.step(
        action="TeleportFull",
        x=pos["x"],
        y=pos["y"],
        z=pos["z"],
        rotation=agent["rotation"],
        horizon=agent["cameraHorizon"],
        standing=agent.get("isStanding", True),
    )


def captioned(image: Image.Image, title: str, subtitle: str, width: int = 360) -> Image.Image:
    image = image.convert("RGB")
    ratio = width / max(1, image.width)
    image = image.resize((width, max(1, int(image.height * ratio))), Image.Resampling.LANCZOS)
    font_title = load_font(20)
    font_body = load_font(15)
    caption_h = 72
    canvas = Image.new("RGB", (image.width, image.height + caption_h), "white")
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    y = image.height + 8
    draw.text((8, y), title, font=font_title, fill=(20, 20, 20))
    draw.text((8, y + 30), subtitle, font=font_body, fill=(55, 55, 55))
    return canvas


def make_panel(
    before: Image.Image,
    pred: Image.Image,
    oracle: Image.Image,
    case: Dict[str, Any],
    pred_gain: int,
    oracle_gain: int,
    output_path: Path,
) -> None:
    panels = [
        captioned(before, "Before", f"visible={int(case['initial_visible_pixels'])}"),
        captioned(pred, f"Model: {case['prediction']}", f"gain={pred_gain}"),
        captioned(oracle, f"Oracle: {case['label']}", f"gain={oracle_gain}"),
    ]
    gap = 18
    header_h = 118
    width = sum(panel.width for panel in panels) + gap * (len(panels) - 1)
    height = header_h + max(panel.height for panel in panels)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(24)
    body_font = load_font(16)
    draw.text(
        (8, 8),
        f"{case['case_group']} / record {case['id']} / {case['scene']} / target={case['object_type']}",
        font=title_font,
        fill=(15, 15, 15),
    )
    draw.text((8, 44), str(case["instruction"]), font=body_font, fill=(45, 45, 45))
    draw.text(
        (8, 72),
        f"Top-1={case['prediction']}, oracle={case['label']}, true_gain(pred)={case['predicted_score']:.1f}",
        font=body_font,
        fill=(45, 45, 45),
    )
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, header_h))
        x += panel.width + gap
    canvas.save(output_path)


def make_gif(frames: List[Image.Image], output_path: Path) -> None:
    resized = [frame.convert("RGB").resize((480, 480), Image.Resampling.LANCZOS) for frame in frames]
    resized[0].save(
        output_path,
        save_all=True,
        append_images=resized[1:],
        duration=[900, 1200, 1200],
        loop=0,
    )


def record_case(controller: Any, cfg: Dict[str, Any], case: Dict[str, Any], record: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_id = record["target"]["object_id"]

    controller.reset(scene=record["scene"])
    before_event = teleport_to_record(controller, record)
    before_pixels = visible_pixels(before_event.instance_masks, target_id)
    save_event(before_event, output_dir, "before")
    before_rgb = Image.fromarray(before_event.frame).convert("RGB")

    pred_event = apply_action(controller, case["prediction"], cfg)
    pred_pixels = visible_pixels(pred_event.instance_masks, target_id)
    pred_success = True if case["prediction"] == "Stay" else bool(pred_event.metadata["lastActionSuccess"])
    pred_gain = pred_pixels - before_pixels if pred_success else -1
    save_event(pred_event, output_dir, "after_prediction")
    pred_rgb = Image.fromarray(pred_event.frame).convert("RGB")

    teleport_to_record(controller, record)
    oracle_event = apply_action(controller, case["label"], cfg)
    oracle_pixels = visible_pixels(oracle_event.instance_masks, target_id)
    oracle_success = True if case["label"] == "Stay" else bool(oracle_event.metadata["lastActionSuccess"])
    oracle_gain = oracle_pixels - before_pixels if oracle_success else -1
    save_event(oracle_event, output_dir, "after_oracle")
    oracle_rgb = Image.fromarray(oracle_event.frame).convert("RGB")

    make_panel(
        before_rgb,
        pred_rgb,
        oracle_rgb,
        case,
        pred_gain=pred_gain,
        oracle_gain=oracle_gain,
        output_path=output_dir / "demo_panel.png",
    )
    make_gif([before_rgb, pred_rgb, oracle_rgb], output_dir / "demo_before_pred_oracle.gif")

    meta = {
        "id": case["id"],
        "case_group": case["case_group"],
        "scene": record["scene"],
        "target": record["target"],
        "instruction": case["instruction"],
        "prediction": case["prediction"],
        "oracle": case["label"],
        "before_pixels": before_pixels,
        "prediction_pixels": pred_pixels,
        "oracle_pixels": oracle_pixels,
        "prediction_gain": pred_gain,
        "oracle_gain": oracle_gain,
        "prediction_success": pred_success,
        "oracle_success": oracle_success,
    }
    (output_dir / "demo_metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, default=Path("runs/case_studies_10k_stay_t200"))
    parser.add_argument("--records", type=Path, default=Path("/mnt/f/see2move/data/ai2thor_oracle_10k_stay/records.jsonl"))
    parser.add_argument("--oracle-config", type=Path, default=Path("configs/ai2thor_oracle.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/qwen_case_demos"))
    parser.add_argument("--case-index", type=int, default=None, help="1-based case index from cases.json.")
    parser.add_argument("--all", action="store_true", help="Render every case from cases.json.")
    args = parser.parse_args()

    from ai2thor.controller import Controller

    cfg = load_config(args.oracle_config)
    cases = json.loads((args.case_dir / "cases.json").read_text(encoding="utf-8"))
    records = {int(record["id"]): record for record in read_jsonl(args.records)}
    if args.all:
        selected = list(enumerate(cases, start=1))
    else:
        idx = args.case_index or 1
        selected = [(idx, cases[idx - 1])]

    controller = Controller(
        scene=selected[0][1]["scene"],
        width=cfg["width"],
        height=cfg["height"],
        gridSize=cfg["grid_size"],
        renderDepthImage=True,
        renderInstanceSegmentation=True,
    )
    results: List[Dict[str, Any]] = []
    try:
        for idx, case in selected:
            record = records[int(case["id"])]
            case_output = args.output_dir / f"{idx:02d}_{case['case_group']}_{case['id']}"
            results.append(record_case(controller, cfg, case, record, case_output))
    finally:
        controller.stop()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "demo_summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "demo_count": len(results)}, indent=2))


if __name__ == "__main__":
    main()
