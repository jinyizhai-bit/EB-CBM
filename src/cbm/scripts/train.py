"""Training entry point.

    python src/cbm/scripts/train.py --config configs/ebcbm.yaml --run_id ebcbm_s0 --gpu 0 \
        [--set train.epochs=2 data.subset=300 ...]

``--gpu`` sets CUDA_VISIBLE_DEVICES unless it is already set (scripts/queue.py sets it) and is recorded in the
run registry.  Configs: configs/*.yaml (``_base_`` inheritance, see cbm.config).
"""
from __future__ import annotations

import argparse
import os
import sys

# this directory contains queue.py, which would shadow the stdlib ``queue`` module: drop it from sys.path
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _here]
sys.path.insert(0, os.path.dirname(os.path.dirname(_here)))   # src/


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--set", nargs="*", default=[], help="key.sub=value overrides (YAML-parsed)")
    a = ap.parse_args(argv)
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    from cbm.config import apply_overrides, load_config
    from cbm.train.trainer import run
    cfg = apply_overrides(load_config(a.config), a.set)
    cfg["_config_file"] = str(a.config)
    cfg["_overrides"] = list(a.set)
    if not cfg.get("run_name"):
        cfg["run_name"] = os.path.splitext(os.path.basename(a.config))[0]
    run(cfg, a.run_id, a.gpu)


if __name__ == "__main__":
    main()
    # persistent DataLoader workers / pin-memory threads can keep the interpreter alive at shutdown; everything
    # (checkpoint, dumps, tracker state, eval.json) is written by now, so exit hard.
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
