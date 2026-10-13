"""Model construction / loading shared by the trainer and the evaluator.

``build_model(cfg, n_classes, concept_names)`` builds ``EBCModel`` unless ``cfg['model']['factory']`` is set to
"package.module:function" (an external model), in which case ``function(cfg, n_classes, concept_names)`` is
called; the returned module must implement the contract ``forward(x, concept_override=None,
mask_override=None) -> dict`` (DESIGN Section 9).
``load_model(run_dir, device)`` rebuilds the model from ``<run_dir>/config.yaml`` and loads ``best.pt``.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import torch
import yaml


def build_model(cfg: dict, n_classes: int, concept_names: list[str]):
    mcfg = cfg["model"]
    fac = mcfg.get("factory")
    if fac:
        mod, fn = fac.split(":")
        return getattr(importlib.import_module(mod), fn)(cfg, n_classes, concept_names)
    from .ebc import EBCModel
    return EBCModel(mcfg, n_classes, concept_names)


def load_model(run_dir: str | Path, device: str = "cuda", ckpt: str = "best.pt"):
    run_dir = Path(run_dir)
    ck = torch.load(run_dir / ckpt, map_location="cpu", weights_only=False)
    cfg = ck.get("config") or yaml.safe_load(open(run_dir / "config.yaml"))
    if "state_dict" in ck and "model" not in ck:
        raise ValueError(f"{run_dir / ckpt}: not an EB-CBM checkpoint (expected keys model/config/class_names/"
                         "concept_names, as written by cbm.train.trainer)")
    cfg["model"].setdefault("backbone", {})["pretrained"] = False   # weights come from the checkpoint
    model = build_model(cfg, len(ck["class_names"]), ck["concept_names"])
    missing, unexpected = model.load_state_dict(ck["model"], strict=False)
    if missing or unexpected:
        print(f"[load_model] missing={missing[:5]}... unexpected={unexpected[:5]}...")
    return model.to(device).eval(), cfg, ck
