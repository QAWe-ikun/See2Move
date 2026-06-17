from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from see2move.ai2thor.oracle_dataset import (
    apply_action,
    choose_best_candidate,
    evaluate_candidates,
    load_config,
    visible_pixels,
)
from see2move.data.ai2thor_records import build_action_vocab, load_depth, pose_features, read_jsonl
from see2move.tools.extract_qwen3vl_features import (
    build_chat_text,
    load_qwen_model,
    move_inputs,
    pool_hidden_state,
)


def depth_feature_stack(path: Path, image_size: int, max_depth: float) -> np.ndarray:
    depth = load_depth(path, image_size=image_size, max_depth=max_depth)[0]
    inverse_depth = 1.0 - depth
    edge = np.zeros_like(depth)
    edge[:, 1:] += np.abs(depth[:, 1:] - depth[:, :-1])
    edge[1:, :] += np.abs(depth[1:, :] - depth[:-1, :])
    edge = np.clip(edge * 4.0, 0.0, 1.0)
    return np.stack([depth, inverse_depth, edge], axis=0).astype(np.float32)


def load_font(size: int) -> ImageFont.ImageFont:
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/mnt/c/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def read_cases(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def depth_preview(depth: np.ndarray) -> Image.Image:
    depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    high = float(np.percentile(depth, 95)) if depth.size else 1.0
    high = max(high, 1.0e-6)
    vis = np.clip(depth / high, 0.0, 1.0)
    return Image.fromarray((vis * 255.0).astype(np.uint8)).convert("RGB")


def save_observation(event: Any, output_dir: Path, step: int) -> Dict[str, Path]:
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    rgb = Image.fromarray(event.frame).convert("RGB")
    depth = np.asarray(event.depth_frame, dtype=np.float32)
    rgb_path = output_dir / f"step_{step:02d}_rgb.png"
    depth_path = output_dir / f"step_{step:02d}_depth.npy"
    depth_vis_path = output_dir / f"step_{step:02d}_depth.png"
    rgb.save(rgb_path)
    np.save(depth_path, depth)
    depth_preview(depth).save(depth_vis_path)
    return {"rgb": rgb_path, "depth": depth_path, "depth_vis": depth_vis_path}


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


def qwen_feature_for_observation(
    processor: Any,
    qwen_model: Any,
    input_device: Any,
    image_path: Path,
    instruction: str,
) -> Any:
    import torch

    image = Image.open(image_path).convert("RGB")
    text = build_chat_text(processor, instruction, image_path)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = move_inputs(inputs, input_device)
    with torch.no_grad():
        outputs = qwen_model(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        pooled = pool_hidden_state(outputs.hidden_states[-1], inputs["attention_mask"])
    return pooled.detach().to(torch.float32).cpu()


def predict_action(
    policy_model: Any,
    ckpt: Dict[str, Any],
    qwen_feature: Any,
    observation_paths: Dict[str, Path],
    event: Any,
    action_vocab: Dict[str, int],
    device: str,
    image_size: int,
    max_depth: float,
    initial_visible_pixels: int,
    stay_threshold: float | None,
    stay_ratio_threshold: float | None,
    stay_ratio_epsilon: float,
) -> Dict[str, Any]:
    import torch
    from see2move.training.train_policy import select_predictions

    idx_to_action = {idx: action for action, idx in action_vocab.items()}
    batch = {
        "qwen_feature": qwen_feature.to(device),
        "depth": torch.from_numpy(
            depth_feature_stack(
                observation_paths["depth"],
                image_size=image_size,
                max_depth=max_depth,
            )
        )
        .unsqueeze(0)
        .to(device),
        "pose": torch.from_numpy(pose_features(event.metadata["agent"]))
        .unsqueeze(0)
        .to(device),
        "initial_visible_pixels": torch.tensor(
            [float(initial_visible_pixels)],
            dtype=torch.float32,
            device=device,
        ),
    }
    train_cfg = ckpt["config"].get("train", {})
    objective = str(train_cfg.get("objective", "classification"))
    score_scale = float(train_cfg.get("score_scale", 1000.0))
    with torch.no_grad():
        logits = policy_model(batch)
        pred_idx = int(
            select_predictions(
                logits,
                batch,
                idx_to_action,
                objective=objective,
                score_scale=score_scale,
                stay_threshold=stay_threshold,
                stay_ratio_threshold=stay_ratio_threshold,
                stay_ratio_epsilon=stay_ratio_epsilon,
            )[0]
            .detach()
            .cpu()
            .item()
        )
    raw_scores = logits[0].detach().cpu()
    predicted_scores = raw_scores * score_scale if objective == "gain_score" else raw_scores
    top_k = min(3, len(idx_to_action))
    top_values, top_indices = predicted_scores.topk(top_k)
    return {
        "action": idx_to_action[pred_idx],
        "predicted_gain": float(predicted_scores[pred_idx].item()),
        "top3": [
            {
                "action": idx_to_action[int(idx.item())],
                "predicted_gain": float(value.item()),
            }
            for value, idx in zip(top_values, top_indices)
        ],
    }


def fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def make_rollout_panel(frames: List[Dict[str, Any]], output_path: Path, target_type: str) -> None:
    cell_w = 250
    cell_h = 190
    caption_h = 60
    row_label_w = 80
    margin = 24
    cols = len(frames)
    width = margin * 2 + row_label_w + cols * cell_w
    height = margin * 2 + 44 + 2 * (cell_h + caption_h)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(22)
    body_font = load_font(15)
    small_font = load_font(13)

    title = frames[0].get("panel_title", f"Greedy rollout target: {target_type}")
    draw.text((margin, margin), title, font=title_font, fill=(20, 20, 20))
    y0 = margin + 44
    row_names = ["RGB", "Depth"]
    for row, name in enumerate(row_names):
        y = y0 + row * (cell_h + caption_h)
        draw.text((margin, y + cell_h // 2 - 10), name, font=body_font, fill=(50, 50, 50))
        for col, frame in enumerate(frames):
            x = margin + row_label_w + col * cell_w
            image_path = frame["rgb"] if row == 0 else frame["depth_vis"]
            canvas.paste(fit_image(Image.open(image_path), (cell_w - 12, cell_h)), (x + 6, y))
            if row == 0:
                caption = frame["caption"]
                draw.text((x + 8, y + cell_h + 8), caption[:34], font=small_font, fill=(25, 25, 25))
                if len(caption) > 34:
                    draw.text((x + 8, y + cell_h + 26), caption[34:68], font=small_font, fill=(25, 25, 25))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def render_video_frame(frame: Dict[str, Any], case_label: str, target_type: str) -> Image.Image:
    width = 960
    height = 540
    canvas = Image.new("RGB", (width, height), (248, 248, 248))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(26)
    body_font = load_font(20)
    small_font = load_font(16)

    draw.text((28, 22), case_label, font=title_font, fill=(20, 20, 20))
    draw.text((28, 58), f"Target: {target_type}", font=body_font, fill=(45, 45, 45))
    draw.text((28, 88), frame["caption"], font=small_font, fill=(65, 65, 65))

    rgb = fit_image(Image.open(frame["rgb"]), (430, 360))
    depth = fit_image(Image.open(frame["depth_vis"]), (430, 360))
    canvas.paste(rgb, (28, 138))
    canvas.paste(depth, (502, 138))
    draw.text((28, 505), "RGB", font=small_font, fill=(45, 45, 45))
    draw.text((502, 505), "Depth", font=small_font, fill=(45, 45, 45))
    return canvas


def write_mp4(frames: List[Image.Image], output_path: Path, fps: int = 2) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = [np.asarray(frame.convert("RGB")) for frame in frames]
    try:
        import imageio.v2 as imageio

        imageio.mimsave(output_path, arrays, fps=fps, macro_block_size=16)
        return
    except Exception as imageio_error:
        try:
            import cv2

            height, width = arrays[0].shape[:2]
            writer = cv2.VideoWriter(
                str(output_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(fps),
                (width, height),
            )
            if not writer.isOpened():
                raise RuntimeError("cv2.VideoWriter failed to open")
            for array in arrays:
                writer.write(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))
            writer.release()
            return
        except Exception as cv2_error:
            raise RuntimeError(
                "Cannot write MP4. Install one backend in the active environment: "
                "pip install imageio imageio-ffmpeg, or pip install opencv-python"
            ) from cv2_error


def parse_case_indices(text: str | None) -> List[int] | None:
    if not text:
        return None
    indices = []
    for item in text.split(","):
        item = item.strip()
        if item:
            indices.append(int(item))
    return indices


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/ai2thor_policy_10k_stay_qwen3vl_gain_score_e25/checkpoint_best.pt"))
    parser.add_argument("--records", type=Path, default=Path("/mnt/f/see2move/data/ai2thor_oracle_10k_stay/records.jsonl"))
    parser.add_argument("--case-dir", type=Path, default=Path("runs/case_studies_10k_stay_t200"))
    parser.add_argument("--case-index", type=int, default=1, help="First 1-based case index when --case-indices is not set.")
    parser.add_argument("--case-indices", default=None, help="Comma-separated 1-based case indices, for example: 1,2,3,4.")
    parser.add_argument("--max-cases", type=int, default=4)
    parser.add_argument("--oracle-config", type=Path, default=Path("configs/ai2thor_oracle.yaml"))
    parser.add_argument("--qwen-model-path", type=Path, default=Path("/mnt/f/models/qwen3_vl"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/qwen_greedy_rollout_demo"))
    parser.add_argument("--video-name", default="greedy_rollout_demo.mp4")
    parser.add_argument("--policy", choices=["model", "oracle", "gt"], default="model")
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--qwen-device", default="auto")
    parser.add_argument("--qwen-dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--stay-threshold", type=float, default=200.0)
    parser.add_argument("--stay-ratio-threshold", type=float, default=None)
    parser.add_argument("--stay-ratio-epsilon", type=float, default=1.0)
    args = parser.parse_args()

    import torch
    from ai2thor.controller import Controller
    from see2move.models.policy import build_qwen_model

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = load_config(args.oracle_config)
    records = {int(record["id"]): record for record in read_jsonl(args.records)}
    cases = read_cases(args.case_dir / "cases.json")
    case_indices = parse_case_indices(args.case_indices)
    if case_indices is None:
        case_indices = list(range(args.case_index, args.case_index + args.max_cases))
    case_indices = [idx for idx in case_indices if 1 <= idx <= len(cases)]
    if not case_indices:
        raise ValueError("No valid case indices selected.")

    ckpt: Dict[str, Any] | None = None
    action_vocab: Dict[str, int] | None = None
    policy_model: Any | None = None
    processor: Any | None = None
    qwen_model: Any | None = None
    input_device: Any | None = None
    image_size = 128
    max_depth = 5.0
    if args.policy == "model":
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        action_vocab = ckpt.get("action_vocab") or build_action_vocab(list(records.values()), ckpt["config"].get("candidate_actions"))
        policy_model = build_qwen_model(
            ckpt["config"],
            qwen_dim=int(ckpt["qwen_feature_dim"]),
            num_actions=len(action_vocab),
        )
        policy_model.load_state_dict(ckpt["model_state"])
        policy_model.to(args.device)
        policy_model.eval()

        processor, qwen_model, input_device = load_qwen_model(
            args.qwen_model_path,
            args.qwen_device,
            args.qwen_dtype,
        )

        data_cfg = ckpt["config"]["data"]
        image_size = int(data_cfg.get("image_size", 128))
        max_depth = float(data_cfg.get("max_depth", 5.0))

    first_case = cases[case_indices[0] - 1]
    first_record = records[int(first_case["id"])]
    controller = Controller(
        scene=first_record["scene"],
        width=cfg["width"],
        height=cfg["height"],
        gridSize=cfg["grid_size"],
        renderDepthImage=True,
        renderInstanceSegmentation=True,
    )
    all_video_frames: List[Image.Image] = []
    summaries: List[Dict[str, Any]] = []
    try:
        for case_index in case_indices:
            case = cases[case_index - 1]
            record = records[int(case["id"])]
            target_id = record["target"]["object_id"]
            instruction = str(record["instruction"])
            target_type = record["target"]["object_type"]
            output_dir = args.output_dir / f"case_{case_index:02d}_{case['case_group']}_{case['id']}_{args.steps}step"
            output_dir.mkdir(parents=True, exist_ok=True)

            controller.reset(scene=record["scene"])
            event = teleport_to_record(controller, record)
            initial_pixels = visible_pixels(event.instance_masks, target_id)
            paths = save_observation(event, output_dir, 0)
            frames: List[Dict[str, Any]] = [
                {
                    **paths,
                    "caption": f"t0 visible={initial_pixels}",
                    "visible_pixels": initial_pixels,
                    "panel_title": f"{args.policy.upper()} greedy rollout target: {target_type}",
                }
            ]
            trace: List[Dict[str, Any]] = []

            for step in range(1, args.steps + 1):
                event = controller.last_event
                before = visible_pixels(event.instance_masks, target_id)
                paths = save_observation(event, output_dir, step - 1)
                candidates = evaluate_candidates(controller, cfg["candidate_actions"], target_id, cfg)
                oracle = choose_best_candidate(candidates)
                if args.policy in {"oracle", "gt"}:
                    action = oracle.action
                    ranked = sorted(candidates, key=lambda item: item.score, reverse=True)
                    prediction = {
                        "action": action,
                        "predicted_gain": float(oracle.score),
                        "top3": [
                            {"action": item.action, "predicted_gain": float(item.score)}
                            for item in ranked[:3]
                        ],
                    }
                else:
                    assert ckpt is not None
                    assert action_vocab is not None
                    assert policy_model is not None
                    assert processor is not None
                    assert qwen_model is not None
                    qwen_feature = qwen_feature_for_observation(
                        processor,
                        qwen_model,
                        input_device,
                        paths["rgb"],
                        instruction,
                    )
                    prediction = predict_action(
                        policy_model,
                        ckpt,
                        qwen_feature,
                        paths,
                        event,
                        action_vocab,
                        args.device,
                        image_size,
                        max_depth,
                        before,
                        args.stay_threshold,
                        args.stay_ratio_threshold,
                        args.stay_ratio_epsilon,
                    )
                    action = prediction["action"]
                next_event = apply_action(controller, action, cfg)
                success = True if action == "Stay" else bool(next_event.metadata["lastActionSuccess"])
                after = visible_pixels(next_event.instance_masks, target_id)
                true_gain = after - before if success else -1
                step_paths = save_observation(next_event, output_dir, step)
                frames.append(
                    {
                        **step_paths,
                        "caption": f"t{step} {action} gain={true_gain} vis={after}",
                        "visible_pixels": after,
                        "panel_title": f"{args.policy.upper()} greedy rollout target: {target_type}",
                    }
                )
                trace.append(
                    {
                        "step": step,
                        "policy": args.policy,
                        "action": action,
                        "predicted_gain": prediction["predicted_gain"],
                        "oracle_action": oracle.action,
                        "oracle_gain": oracle.score,
                        "before_pixels": before,
                        "after_pixels": after,
                        "true_gain": true_gain,
                        "success": success,
                        "top3": prediction["top3"],
                    }
                )

            summary = {
                "case_index": case_index,
                "record_id": case["id"],
                "scene": record["scene"],
                "target": record["target"],
                "instruction": instruction,
                "policy": args.policy,
                "steps": args.steps,
                "initial_pixels": initial_pixels,
                "final_pixels": frames[-1]["visible_pixels"],
                "total_gain": frames[-1]["visible_pixels"] - initial_pixels,
                "trace": trace,
            }
            summaries.append(summary)
            (output_dir / "rollout_trace.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            make_rollout_panel(frames, output_dir / "rollout_panel.png", target_type)

            case_label = f"Case {case_index}: {case['case_group']} / record {case['id']}"
            for frame in frames:
                all_video_frames.append(render_video_frame(frame, case_label, target_type))
            if case_index != case_indices[-1]:
                all_video_frames.extend([all_video_frames[-1]] * max(1, args.fps))
    finally:
        controller.stop()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.output_dir / args.video_name
    write_mp4(all_video_frames, video_path, fps=args.fps)
    summary_path = args.output_dir / "greedy_rollout_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "video": str(video_path),
                "summary": str(summary_path),
                "case_count": len(summaries),
                "steps": args.steps,
                "total_gains": [item["total_gain"] for item in summaries],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
