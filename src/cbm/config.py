"""YAML config system: ``_base_`` inheritance + ``--set a.b.c=value`` overrides (values parsed as YAML)."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

from cbm.paths import CONFIG_DIR


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists() and (CONFIG_DIR / path).exists():
        path = CONFIG_DIR / path
    cfg = yaml.safe_load(open(path)) or {}
    base = cfg.pop("_base_", None)
    if base:
        bp = (path.parent / base) if not Path(base).is_absolute() else Path(base)
        cfg = deep_update(load_config(bp), cfg)
    return cfg


def apply_overrides(cfg: dict, sets: list[str] | None) -> dict:
    cfg = copy.deepcopy(cfg)
    for s in sets or []:
        if "=" not in s:
            raise ValueError(f"--set expects key=value, got {s!r}")
        k, v = s.split("=", 1)
        val = yaml.safe_load(v)
        d = cfg
        parts = k.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val
    return cfg
