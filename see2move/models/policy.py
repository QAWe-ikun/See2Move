from __future__ import annotations

from typing import Dict, Any

import torch
from torch import nn


class See2MovePolicy(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_actions: int,
        text_dim: int = 64,
        hidden_dim: int = 128,
        pose_dim: int = 7,
    ) -> None:
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.text_embedding = nn.Embedding(vocab_size, text_dim, padding_idx=0)
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, 32),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + text_dim + 32, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_actions),
        )

    def encode_text(self, text_ids: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        embeddings = self.text_embedding(text_ids)
        mask = text_mask.unsqueeze(-1)
        summed = (embeddings * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        image_feat = self.image_encoder(batch["image"])
        text_feat = self.encode_text(batch["text_ids"], batch["text_mask"])
        pose_feat = self.pose_encoder(batch["pose"])
        return self.head(torch.cat([image_feat, text_feat, pose_feat], dim=1))


def build_model(config: Dict[str, Any], vocab_size: int, num_actions: int) -> See2MovePolicy:
    model_cfg = config.get("model", {})
    return See2MovePolicy(
        vocab_size=vocab_size,
        num_actions=num_actions,
        text_dim=int(model_cfg.get("text_dim", 64)),
        hidden_dim=int(model_cfg.get("hidden_dim", 128)),
    )
