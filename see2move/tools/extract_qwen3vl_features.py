from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

def check_qwen_dependencies() -> None:
    try:
        import torchvision  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Qwen3-VL feature extraction requires torchvision. "
            "Install it in the active environment with: pip install torchvision"
        ) from exc


def load_qwen_model(model_path: Path, device: str, dtype_name: str) -> tuple[Any, Any, torch.device]:
    import torch

    check_qwen_dependencies()
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText
        model_cls = AutoModelForImageTextToText
    except ImportError:
        from transformers import AutoModelForCausalLM
        model_cls = AutoModelForCausalLM

    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[dtype_name]
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": True,
    }
    if dtype != "auto":
        model_kwargs["torch_dtype"] = dtype

    if device == "auto":
        model_kwargs["device_map"] = "auto"
        model = model_cls.from_pretrained(model_path, **model_kwargs)
        input_device = next(model.parameters()).device
    else:
        input_device = torch.device(device)
        model = model_cls.from_pretrained(model_path, **model_kwargs).to(input_device)

    model.eval()
    return processor, model, input_device


def build_chat_text(processor: Any, instruction: str, image_path: Path) -> str:
    prompt = (
        "Encode this embodied navigation observation. "
        "Focus on objects, occlusions, free space, and where the camera should move next.\n"
        f"Instruction: {instruction}"
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def move_inputs(inputs: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    import torch

    moved = {}
    for key, value in inputs.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def pool_hidden_state(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(hidden.device).unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def load_partial(path: Path) -> Dict[str, Any] | None:
    import torch

    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


def save_features(path: Path, payload: Dict[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def extract_features(args: argparse.Namespace) -> None:
    import torch
    from PIL import Image
    from tqdm.auto import tqdm

    from see2move.data.ai2thor_records import read_jsonl, record_text

    records = read_jsonl(args.records)
    if args.max_records is not None:
        records = records[: args.max_records]

    partial_path = args.output.with_suffix(args.output.suffix + ".partial")
    existing = load_partial(partial_path) if args.resume else None
    features: List[torch.Tensor] = []
    record_ids: List[int] = []
    start = 0
    if existing is not None:
        features = [tensor for tensor in existing["features"]]
        record_ids = [int(item) for item in existing["record_ids"]]
        start = len(record_ids)

    processor, model, input_device = load_qwen_model(args.model_path, args.device, args.dtype)
    progress = tqdm(total=len(records), initial=start, desc="Extracting Qwen3-VL features")
    try:
        for offset in range(start, len(records), args.batch_size):
            batch_records = records[offset : offset + args.batch_size]
            images = [
                Image.open(args.data_dir / record["rgb"]).convert("RGB")
                for record in batch_records
            ]
            texts = [
                build_chat_text(processor, record_text(record), args.data_dir / record["rgb"])
                for record in batch_records
            ]
            inputs = processor(
                text=texts,
                images=images,
                padding=True,
                return_tensors="pt",
            )
            inputs = move_inputs(inputs, input_device)
            with torch.no_grad():
                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                pooled = pool_hidden_state(outputs.hidden_states[-1], inputs["attention_mask"])
                pooled = pooled.detach().to(torch.float16).cpu()

            for record, feature in zip(batch_records, pooled):
                features.append(feature)
                record_ids.append(int(record.get("id", len(record_ids))))

            progress.update(len(batch_records))
            if len(record_ids) % args.save_every == 0 or len(record_ids) >= len(records):
                save_features(
                    partial_path,
                    {
                        "features": features,
                        "record_ids": record_ids,
                        "records_path": str(args.records),
                        "data_dir": str(args.data_dir),
                        "model_path": str(args.model_path),
                    },
                )
    finally:
        progress.close()

    feature_tensor = torch.stack(features, dim=0).contiguous()
    save_features(
        args.output,
        {
            "features": feature_tensor,
            "record_ids": record_ids,
            "records_path": str(args.records),
            "data_dir": str(args.data_dir),
            "model_path": str(args.model_path),
            "feature_dim": int(feature_tensor.shape[1]),
        },
    )
    if partial_path.exists():
        partial_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=Path("/mnt/f/see2move/data/ai2thor_oracle_10k_stay/records.jsonl"))
    parser.add_argument("--data-dir", type=Path, default=Path("/mnt/f/see2move/data/ai2thor_oracle_10k_stay"))
    parser.add_argument("--model-path", type=Path, default=Path("/mnt/f/models/qwen3_vl"))
    parser.add_argument("--output", type=Path, default=Path("/mnt/f/see2move/features/qwen3vl_ai2thor_oracle_10k_stay.pt"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--resume", action="store_true", default=True)
    args = parser.parse_args()
    extract_features(args)


if __name__ == "__main__":
    main()
