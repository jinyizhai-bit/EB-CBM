"""Shared backbones.  Common interface: ``forward(x) -> F`` with F of shape (N, d, h, w).

* ``DinoV2Backbone``  - HF ``facebook/dinov2-with-registers-base`` (ViT-B/14 + 4 registers).
  At 448x448 input the patch grid is 32x32; CLS and register tokens are dropped, the final-layer-norm
  patch tokens are reshaped to (N, 768, 32, 32).  ``self.last_cls`` keeps the CLS token of the last call
  (not part of the interface; used by nobody in the EBC models, handy for baselines).
* ``ResNetFPNBackbone`` - timm ResNet-50 (ImageNet), layer3 (stride 16) + layer4 (stride 32) fused by a
  1-level FPN into a (N, 256, 28, 28) map at 448 input.

``build_backbone(cfg)`` takes the ``model.backbone`` config dict.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoV2Backbone(nn.Module):
    def __init__(self, name: str = "facebook/dinov2-with-registers-base", pretrained: bool = True,
                 grad_checkpointing: bool = False, freeze: bool = False):
        super().__init__()
        from transformers import AutoConfig, AutoModel
        if pretrained:
            self.vit = AutoModel.from_pretrained(name, attn_implementation="sdpa")
        else:
            self.vit = AutoModel.from_config(AutoConfig.from_pretrained(name), attn_implementation="sdpa")
        cfg = self.vit.config
        self.patch = cfg.patch_size
        self.n_prefix = 1 + int(getattr(cfg, "num_register_tokens", 0))
        self.out_dim = cfg.hidden_size
        if grad_checkpointing:
            self.vit.gradient_checkpointing_enable()
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad_(False)
        self.last_cls = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, H, W = x.shape
        h, w = H // self.patch, W // self.patch
        tok = self.vit(pixel_values=x).last_hidden_state          # (N, prefix + h*w, d)
        self.last_cls = tok[:, 0]
        pt = tok[:, self.n_prefix:]
        return pt.transpose(1, 2).reshape(n, self.out_dim, h, w)


class ResNetFPNBackbone(nn.Module):
    def __init__(self, name: str = "resnet50.a1_in1k", pretrained: bool = True, out_dim: int = 256,
                 freeze: bool = False, **_):
        super().__init__()
        import timm
        self.body = timm.create_model(name, pretrained=pretrained, features_only=True, out_indices=(3, 4))
        c3, c4 = self.body.feature_info.channels()
        self.lat3 = nn.Conv2d(c3, out_dim, 1)
        self.lat4 = nn.Conv2d(c4, out_dim, 1)
        self.smooth = nn.Sequential(nn.Conv2d(out_dim, out_dim, 3, padding=1), nn.GroupNorm(32, out_dim), nn.GELU())
        self.out_dim = out_dim
        if freeze:
            for p in self.body.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f3, f4 = self.body(x)
        p = self.lat3(f3) + F.interpolate(self.lat4(f4), size=f3.shape[-2:], mode="bilinear", align_corners=False)
        return self.smooth(p)


def build_backbone(cfg: dict) -> nn.Module:
    kind = cfg.get("type", "dinov2")
    if kind == "dinov2":
        return DinoV2Backbone(cfg.get("name", "facebook/dinov2-with-registers-base"), cfg.get("pretrained", True),
                              cfg.get("grad_checkpointing", False), cfg.get("freeze", False))
    if kind in ("resnet50_fpn", "resnet_fpn"):
        return ResNetFPNBackbone(cfg.get("name", "resnet50.a1_in1k"), cfg.get("pretrained", True),
                                 cfg.get("out_dim", 256), cfg.get("freeze", False))
    raise ValueError(f"unknown backbone type {kind!r}")
