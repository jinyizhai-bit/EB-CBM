"""Repository paths used by every module (override with environment variables).

CBM_ROOT       repository root; default: the first ancestor of this file that contains ``configs/`` and ``src/``
CBM_DATA_ROOT  data root with ``raw/``, ``processed/``, ``metadata/`` and ``splits/``; default: <CBM_ROOT>/data
CBM_LAMA_PT    TorchScript LaMa checkpoint for the counterfactual edits; default: <CBM_ROOT>/models/big-lama.pt
"""
from __future__ import annotations

import os
from pathlib import Path


def _find_root() -> Path:
    env = os.environ.get("CBM_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    for p in Path(__file__).resolve().parents:
        if (p / "configs").is_dir() and (p / "src").is_dir():
            return p
    return Path.cwd()


ROOT = _find_root()
SRC = ROOT / "src"
CONFIG_DIR = ROOT / "configs"
DATA_ROOT = Path(os.environ.get("CBM_DATA_ROOT") or ROOT / "data").expanduser()
RAW = DATA_ROOT / "raw"                     # downloaded archives, unpacked in place
PROCESSED = DATA_ROOT / "processed"         # manifests + 512-px caches (built by cbm.data.manifests)
METADATA = DATA_ROOT / "metadata"           # ISIC Archive metadata snapshots (shipped with the repository)
SPLITS = DATA_ROOT / "splits"               # fixed validation hold-out ids (shipped with the repository)
EXPERIMENTS = ROOT / "experiments"
RUNS_DIR = EXPERIMENTS / "runs"
STATE_DIR = EXPERIMENTS / "state"
FIG_ROOT = ROOT / "figures"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
LAMA_PT = Path(os.environ.get("CBM_LAMA_PT") or MODELS_DIR / "big-lama.pt").expanduser()
CF_DIR = PROCESSED / "isic2018_task12" / "cf_512"
CF_INDEX = CF_DIR / "index.csv"
