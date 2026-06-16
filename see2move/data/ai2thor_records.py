from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

TOKEN_RE = re.compile(r"[a-z0-9]+")
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def use_modality(ablation: Dict[str, Any] | None, name: str) -> bool:
    if not ablation:
        return True
    return bool(ablation.get(f"use_{name}", True))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def record_text(record: Dict[str, Any]) -> str:
    history = record.get("history", "")
    if isinstance(history, list):
        history_text = " ".join(str(item) for item in history)
    else:
        history_text = str(history)
    if history_text:
        return f"{record['instruction']} {history_text}"
    return record["instruction"]


def build_text_vocab(records: Iterable[Dict[str, Any]], max_size: int = 4096) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for record in records:
        counts.update(tokenize(record_text(record)))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, _ in counts.most_common(max(0, max_size - len(vocab))):
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def encode_text(text: str, vocab: Dict[str, int], max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokenize(text)[:max_len]]
    length = len(ids)
    if length < max_len:
        ids += [vocab[PAD_TOKEN]] * (max_len - length)
    mask = [1.0] * length + [0.0] * (max_len - length)
    return np.asarray(ids, dtype=np.int64), np.asarray(mask, dtype=np.float32)


def build_action_vocab(records: Iterable[Dict[str, Any]], fallback_actions: Sequence[str] | None = None) -> Dict[str, int]:
    actions = list(fallback_actions or [])
    seen = set(actions)
    for record in records:
        label = record["label"]
        if label not in seen:
            actions.append(label)
            seen.add(label)
        for candidate in record.get("candidates", []):
            action = candidate["action"]
            if action not in seen:
                actions.append(action)
                seen.add(action)
    return {action: idx for idx, action in enumerate(actions)}


def load_rgb(path: Path, image_size: int) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))


def load_depth(path: Path, image_size: int, max_depth: float) -> np.ndarray:
    depth = np.load(path).astype(np.float32)
    depth = np.nan_to_num(depth, nan=0.0, posinf=max_depth, neginf=0.0)
    depth = np.clip(depth, 0.0, max_depth) / max_depth
    image = Image.fromarray((depth * 255.0).astype(np.uint8)).resize(
        (image_size, image_size), Image.BILINEAR
    )
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return arr[None, :, :]


def pose_features(agent: Dict[str, Any]) -> np.ndarray:
    position = agent.get("position", {})
    rotation = agent.get("rotation", {})
    yaw = math.radians(float(rotation.get("y", 0.0)))
    horizon = float(agent.get("cameraHorizon", 0.0))
    standing = 1.0 if agent.get("isStanding", True) else 0.0
    return np.asarray(
        [
            float(position.get("x", 0.0)),
            float(position.get("y", 0.0)),
            float(position.get("z", 0.0)),
            math.sin(yaw),
            math.cos(yaw),
            horizon / 90.0,
            standing,
        ],
        dtype=np.float32,
    )


def candidate_score_features(
    record: Dict[str, Any],
    action_vocab: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    scores = np.zeros(len(action_vocab), dtype=np.float32)
    mask = np.zeros(len(action_vocab), dtype=np.float32)
    for candidate in record.get("candidates", []):
        action = candidate.get("action")
        if action not in action_vocab:
            continue
        idx = action_vocab[action]
        scores[idx] = float(candidate.get("score", 0.0))
        mask[idx] = 1.0
    return scores, mask


def split_records(records: Sequence[Dict[str, Any]], val_fraction: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)
    val_count = int(round(len(records) * val_fraction))
    val_indices = set(indices[:val_count].tolist())
    train, val = [], []
    for idx, record in enumerate(records):
        if idx in val_indices:
            val.append(record)
        else:
            train.append(record)
    return train, val


def split_records_by_scene(
    records: Sequence[Dict[str, Any]],
    val_fraction: float,
    seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    scene_names = sorted({str(record.get("scene", "")) for record in records})
    if len(scene_names) <= 1 or val_fraction <= 0:
        return list(records), [], []

    rng = np.random.default_rng(seed)
    scene_indices = np.arange(len(scene_names))
    rng.shuffle(scene_indices)
    val_scene_count = int(round(len(scene_names) * val_fraction))
    val_scene_count = max(1, min(len(scene_names) - 1, val_scene_count))
    val_scenes = {scene_names[idx] for idx in scene_indices[:val_scene_count]}

    train, val = [], []
    for record in records:
        if str(record.get("scene", "")) in val_scenes:
            val.append(record)
        else:
            train.append(record)
    return train, val, sorted(val_scenes)


class AI2ThorOracleDataset:
    def __init__(
        self,
        records: Sequence[Dict[str, Any]],
        data_dir: Path,
        text_vocab: Dict[str, int],
        action_vocab: Dict[str, int],
        image_size: int,
        max_depth: float,
        max_text_len: int,
        ablation: Dict[str, Any] | None = None,
    ) -> None:
        self.records = list(records)
        self.data_dir = data_dir
        self.text_vocab = text_vocab
        self.action_vocab = action_vocab
        self.image_size = image_size
        self.max_depth = max_depth
        self.max_text_len = max_text_len
        self.ablation = ablation or {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        import torch

        record = self.records[index]
        rgb = load_rgb(self.data_dir / record["rgb"], self.image_size)
        depth = load_depth(self.data_dir / record["depth"], self.image_size, self.max_depth)
        text_ids, text_mask = encode_text(record_text(record), self.text_vocab, self.max_text_len)
        pose = pose_features(record["agent"])
        if not use_modality(self.ablation, "rgb"):
            rgb = np.zeros_like(rgb)
        if not use_modality(self.ablation, "depth"):
            depth = np.zeros_like(depth)
        if not use_modality(self.ablation, "text"):
            text_ids = np.zeros_like(text_ids)
            text_mask = np.zeros_like(text_mask)
        if not use_modality(self.ablation, "pose"):
            pose = np.zeros_like(pose)
        label = self.action_vocab[record["label"]]
        candidate_scores, candidate_mask = candidate_score_features(record, self.action_vocab)
        return {
            "record_id": torch.tensor(int(record.get("id", index)), dtype=torch.long),
            "image": torch.from_numpy(np.concatenate([rgb, depth], axis=0)),
            "text_ids": torch.from_numpy(text_ids),
            "text_mask": torch.from_numpy(text_mask),
            "pose": torch.from_numpy(pose),
            "candidate_scores": torch.from_numpy(candidate_scores),
            "candidate_mask": torch.from_numpy(candidate_mask),
            "initial_visible_pixels": torch.tensor(
                float(record.get("initial_visible_pixels", 0.0)),
                dtype=torch.float32,
            ),
            "label": torch.tensor(label, dtype=torch.long),
        }
