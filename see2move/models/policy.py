from __future__ import annotations

from typing import Dict, Any

import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(8, out_channels),
            )
        else:
            self.skip = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv2(self.conv1(x)) + self.skip(x))


class See2MovePolicy(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_actions: int,
        text_dim: int = 64,
        hidden_dim: int = 128,
        pose_dim: int = 7,
        dropout: float = 0.1,
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
            nn.Dropout(dropout),
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


class ResidualImageEncoder(nn.Module):
    def __init__(self, in_channels: int = 4, base_channels: int = 48, output_dim: int = 256) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, c1, stride=2),
            ResidualBlock(c1, c1),
            ResidualBlock(c1, c2, stride=2),
            ResidualBlock(c2, c2),
            ResidualBlock(c2, c3, stride=2),
            ResidualBlock(c3, c3),
            ResidualBlock(c3, c3),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.projection = nn.Sequential(
            nn.Linear(c3, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.projection(self.encoder(image))


class TextGRUEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, text_ids: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        lengths = text_mask.sum(dim=1).clamp_min(1).long().cpu()
        embeddings = self.embedding(text_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        final_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.projection(final_hidden)


class ActionScoringHead(nn.Module):
    def __init__(self, state_dim: int, num_actions: int, dropout: float) -> None:
        super().__init__()
        self.state = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Dropout(dropout),
            nn.Linear(state_dim, state_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(state_dim),
        )
        self.action_embedding = nn.Embedding(num_actions, state_dim)
        self.action_bias = nn.Parameter(torch.zeros(num_actions))
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        state = self.state(fused)
        state = F.normalize(state, dim=1)
        actions = F.normalize(self.action_embedding.weight, dim=1)
        scale = self.logit_scale.exp().clamp(max=20.0)
        return scale * (state @ actions.t()) + self.action_bias


class ResidualFusionPolicy(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_actions: int,
        text_dim: int = 128,
        text_hidden_dim: int = 96,
        image_dim: int = 256,
        image_base_channels: int = 48,
        pose_dim: int = 7,
        pose_hidden_dim: int = 64,
        fusion_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.image_encoder = ResidualImageEncoder(
            in_channels=4,
            base_channels=image_base_channels,
            output_dim=image_dim,
        )
        self.text_encoder = TextGRUEncoder(
            vocab_size=vocab_size,
            embedding_dim=text_dim,
            hidden_dim=text_hidden_dim,
            output_dim=text_dim,
            dropout=dropout,
        )
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, pose_hidden_dim),
            nn.LayerNorm(pose_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(pose_hidden_dim, pose_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.image_gate = nn.Sequential(
            nn.Linear(text_dim + pose_hidden_dim, image_dim),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(image_dim + text_dim + pose_hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.ReLU(inplace=True),
        )
        self.head = ActionScoringHead(fusion_dim, num_actions, dropout=dropout)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        image_feat = self.image_encoder(batch["image"])
        text_feat = self.text_encoder(batch["text_ids"], batch["text_mask"])
        pose_feat = self.pose_encoder(batch["pose"])
        image_feat = image_feat * self.image_gate(torch.cat([text_feat, pose_feat], dim=1))
        fused = self.fusion(torch.cat([image_feat, text_feat, pose_feat], dim=1))
        return self.head(fused)


class DepthFeatureEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, output_dim: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 32, stride=2),
            ResidualBlock(32, 32),
            ResidualBlock(32, 64, stride=2),
            ResidualBlock(64, 64),
            ResidualBlock(64, 128, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        return self.encoder(depth)


class QwenDepthPolicy(nn.Module):
    def __init__(
        self,
        qwen_dim: int,
        num_actions: int,
        qwen_hidden_dim: int = 512,
        depth_dim: int = 128,
        pose_dim: int = 7,
        pose_hidden_dim: int = 64,
        hidden_dim: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.qwen_encoder = nn.Sequential(
            nn.LayerNorm(qwen_dim),
            nn.Linear(qwen_dim, qwen_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.depth_encoder = DepthFeatureEncoder(in_channels=3, output_dim=depth_dim)
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, pose_hidden_dim),
            nn.LayerNorm(pose_hidden_dim),
            nn.ReLU(inplace=True),
        )
        fusion_dim = qwen_hidden_dim + depth_dim + pose_hidden_dim
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        qwen_feat = self.qwen_encoder(batch["qwen_feature"].float())
        depth_feat = self.depth_encoder(batch["depth"])
        pose_feat = self.pose_encoder(batch["pose"])
        return self.head(torch.cat([qwen_feat, depth_feat, pose_feat], dim=1))


class QwenGatedDepthPolicy(nn.Module):
    def __init__(
        self,
        qwen_dim: int,
        num_actions: int,
        qwen_hidden_dim: int = 512,
        depth_dim: int = 128,
        pose_dim: int = 7,
        pose_hidden_dim: int = 64,
        hidden_dim: int = 512,
        dropout: float = 0.2,
        modality_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.modality_dropout = modality_dropout
        self.qwen_encoder = nn.Sequential(
            nn.LayerNorm(qwen_dim),
            nn.Linear(qwen_dim, qwen_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.depth_encoder = DepthFeatureEncoder(in_channels=3, output_dim=depth_dim)
        self.pose_encoder = nn.Sequential(
            nn.Linear(pose_dim, pose_hidden_dim),
            nn.LayerNorm(pose_hidden_dim),
            nn.ReLU(inplace=True),
        )
        fusion_dim = qwen_hidden_dim + depth_dim + pose_hidden_dim
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, fusion_dim),
            nn.Sigmoid(),
        )
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_actions),
        )

    def drop_modality(self, feature: torch.Tensor) -> torch.Tensor:
        if not self.training or self.modality_dropout <= 0:
            return feature
        keep = torch.rand(feature.shape[0], 1, device=feature.device) >= self.modality_dropout
        return feature * keep.to(feature.dtype)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        qwen_feat = self.drop_modality(self.qwen_encoder(batch["qwen_feature"].float()))
        depth_feat = self.drop_modality(self.depth_encoder(batch["depth"]))
        pose_feat = self.drop_modality(self.pose_encoder(batch["pose"]))
        fused = torch.cat([qwen_feat, depth_feat, pose_feat], dim=1)
        return self.head(fused * self.gate(fused))


def build_qwen_model(config: Dict[str, Any], qwen_dim: int, num_actions: int) -> nn.Module:
    model_cfg = config.get("model", {})
    architecture = str(model_cfg.get("architecture", "concat"))
    kwargs = {
        "qwen_dim": qwen_dim,
        "num_actions": num_actions,
        "qwen_hidden_dim": int(model_cfg.get("qwen_hidden_dim", 512)),
        "depth_dim": int(model_cfg.get("depth_dim", 128)),
        "pose_hidden_dim": int(model_cfg.get("pose_hidden_dim", 64)),
        "hidden_dim": int(model_cfg.get("hidden_dim", 512)),
        "dropout": float(model_cfg.get("dropout", 0.2)),
    }
    if architecture in {"concat", "qwen_depth"}:
        return QwenDepthPolicy(**kwargs)
    if architecture == "gated":
        return QwenGatedDepthPolicy(
            **kwargs,
            modality_dropout=float(model_cfg.get("modality_dropout", 0.0)),
        )
    raise ValueError(f"Unsupported Qwen architecture: {architecture}")


def build_model(config: Dict[str, Any], vocab_size: int, num_actions: int) -> nn.Module:
    model_cfg = config.get("model", {})
    architecture = str(model_cfg.get("architecture", "simple"))
    if architecture == "simple":
        return See2MovePolicy(
            vocab_size=vocab_size,
            num_actions=num_actions,
            text_dim=int(model_cfg.get("text_dim", 64)),
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
    if architecture == "residual_fusion":
        return ResidualFusionPolicy(
            vocab_size=vocab_size,
            num_actions=num_actions,
            text_dim=int(model_cfg.get("text_dim", 128)),
            text_hidden_dim=int(model_cfg.get("text_hidden_dim", 96)),
            image_dim=int(model_cfg.get("image_dim", 256)),
            image_base_channels=int(model_cfg.get("image_base_channels", 48)),
            pose_hidden_dim=int(model_cfg.get("pose_hidden_dim", 64)),
            fusion_dim=int(model_cfg.get("fusion_dim", 256)),
            hidden_dim=int(model_cfg.get("hidden_dim", 256)),
            dropout=float(model_cfg.get("dropout", 0.2)),
        )
    raise ValueError(f"Unsupported model architecture: {architecture}")
