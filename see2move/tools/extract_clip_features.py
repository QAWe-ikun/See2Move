from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List


def load_existing(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    import torch

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload.get("features"), list):
        payload["features"] = [tensor for tensor in payload["features"]]
    return payload


def save_payload(path: Path, payload: Dict[str, Any]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def normalize_features(tensor: Any) -> Any:
    import torch.nn.functional as F

    return F.normalize(tensor.float(), dim=1)


def load_model_and_processor(args: argparse.Namespace) -> tuple[Any, Any]:
    try:
        import sentencepiece  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "SigLIP feature extraction requires sentencepiece. Install it in the active "
            "environment with: pip install sentencepiece"
        ) from exc

    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=args.trust_remote_code,
    )
    return processor, model


def output_embeddings(outputs: Any) -> tuple[Any, Any]:
    if not hasattr(outputs, "image_embeds") or not hasattr(outputs, "text_embeds"):
        raise AttributeError(
            "The selected model did not return image_embeds/text_embeds. "
            "Use a CLIP/SigLIP-style model or add a model-specific pooling adapter."
        )
    return outputs.image_embeds, outputs.text_embeds


def extract_features(args: argparse.Namespace) -> None:
    import torch
    from PIL import Image
    from tqdm.auto import tqdm

    from see2move.data.ai2thor_records import read_jsonl

    records = read_jsonl(args.records)
    existing = load_existing(args.output) if args.resume else None
    if existing is not None:
        features: List[Any] = list(existing["features"])
        record_ids: List[int] = [int(record_id) for record_id in existing["record_ids"]]
    else:
        features = []
        record_ids = []

    start = len(features)
    if start > len(records):
        raise ValueError(f"Existing feature file has {start} entries but records only has {len(records)}.")

    processor, model = load_model_and_processor(args)
    if args.dtype == "float16":
        model = model.half()
    elif args.dtype == "bfloat16":
        model = model.to(dtype=torch.bfloat16)
    model.to(args.device)
    model.eval()

    progress = tqdm(total=len(records), initial=start, desc="Extracting SigLIP features")
    with torch.no_grad():
        for offset in range(start, len(records), args.batch_size):
            batch_records = records[offset : offset + args.batch_size]
            images = [
                Image.open(args.data_dir / str(record["rgb"])).convert("RGB")
                for record in batch_records
            ]
            texts = [str(record["instruction"]) for record in batch_records]
            inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
            inputs = {
                key: value.to(args.device)
                for key, value in inputs.items()
            }
            if args.dtype == "float16" and "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].half()
            elif args.dtype == "bfloat16" and "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch.bfloat16)

            outputs = model(**inputs)
            image_embeds, text_embeds = output_embeddings(outputs)
            image_feat = normalize_features(image_embeds)
            text_feat = normalize_features(text_embeds)
            combined = torch.cat(
                [
                    image_feat,
                    text_feat,
                    image_feat * text_feat,
                    torch.abs(image_feat - text_feat),
                ],
                dim=1,
            )
            combined = combined.detach().to(torch.float16).cpu()
            for record, feature in zip(batch_records, combined):
                record_ids.append(int(record.get("id", offset)))
                features.append(feature)
            progress.update(len(batch_records))

            if args.save_every > 0 and len(features) % args.save_every == 0:
                save_payload(
                    args.output,
                    {
                        "record_ids": record_ids,
                        "features": features,
                        "feature_type": "siglip_concat_image_text_product_absdiff",
                        "model_path": str(args.model_path),
                    },
                )
    progress.close()

    save_payload(
        args.output,
        {
            "record_ids": record_ids,
            "features": torch.stack(features, dim=0).contiguous(),
            "feature_type": "siglip_concat_image_text_product_absdiff",
            "model_path": str(args.model_path),
        },
    )
    print({"output": str(args.output), "records": len(record_ids), "feature_dim": int(features[0].numel()) if features else 0})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=Path("/mnt/f/see2move/data/ai2thor_oracle_10k_stay/records.jsonl"))
    parser.add_argument("--data-dir", type=Path, default=Path("/mnt/f/see2move/data/ai2thor_oracle_10k_stay"))
    parser.add_argument("--output", type=Path, default=Path("/mnt/f/see2move/features/siglip_so400m_ai2thor_oracle_10k_stay.pt"))
    parser.add_argument("--model-path", default="/mnt/e/project/SceneReVis/ckpt/google/siglip-so400m-patch14-384")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--save-every", type=int, default=256)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    import torch

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device == "cpu" and args.dtype != "float32":
        args.dtype = "float32"
    extract_features(args)


if __name__ == "__main__":
    main()
