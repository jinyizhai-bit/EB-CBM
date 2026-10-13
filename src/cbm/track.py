"""Lightweight JSON run registry (file based, no external service).

State lives under ``experiments/state`` (``runs.json``, ``results.json``, ``figures.json``, ``events.jsonl``);
per-run metrics go to ``experiments/runs/<run_id>/metrics.jsonl`` (one JSON record per line:
``{"ts", "epoch", "step", "split": "train|val|test", <metric>: value, ...}``).  Writes are atomic (temp file +
rename) and guarded by an fcntl lock, so several training processes may update the registry concurrently.
"""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from cbm.paths import ROOT, RUNS_DIR, STATE_DIR  # noqa: F401  (ROOT is re-exported)

STATE_DIR.mkdir(parents=True, exist_ok=True)
RUNS_DIR.mkdir(parents=True, exist_ok=True)
_LOCK = STATE_DIR / ".lock"


class _Lock:
    def __enter__(self):
        self.f = open(_LOCK, "w")
        fcntl.flock(self.f, fcntl.LOCK_EX)
        return self

    def __exit__(self, *a):
        fcntl.flock(self.f, fcntl.LOCK_UN)
        self.f.close()


def _read(name: str, default: Any) -> Any:
    p = STATE_DIR / name
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text())
    except Exception:
        return default


def _write(name: str, obj: Any) -> None:
    obj["updated"] = time.time()
    p = STATE_DIR / name
    fd, tmp = tempfile.mkstemp(dir=STATE_DIR, prefix=f".{name}.")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, p)


def _upsert(name: str, key: str, item: dict, id_field: str = "id") -> dict:
    with _Lock():
        d = _read(name, {key: []})
        items = d.setdefault(key, [])
        for i, it in enumerate(items):
            if it.get(id_field) == item[id_field]:
                merged = {**it, **item}
                items[i] = merged
                _write(name, d)
                return merged
        items.append(item)
        _write(name, d)
        return item


# ---------------------------------------------------------------- events
def log_event(msg: str, level: str = "info", source: str = "pipeline") -> None:
    rec = {"ts": time.time(), "level": level, "source": source, "msg": msg}
    with _Lock():
        with open(STATE_DIR / "events.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------- runs
def register_run(run_id: str, name: str, idea: str, gpu: int, total_epochs: int, config: dict | None = None) -> Path:
    d = RUNS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    _upsert("runs.json", "runs", {
        "id": run_id, "name": name, "idea": idea, "gpu": gpu, "status": "running", "epoch": 0,
        "total_epochs": total_epochs, "started": time.time(), "ended": None,
        "log": str(d / "metrics.jsonl"), "best": {}, "config": config or {}, "pid": os.getpid()})
    log_event(f"run {run_id} ({name}) started on GPU{gpu}", "info", source=f"run:{run_id}")
    return d


def log_metrics(run_id: str, split: str, epoch: int, step: int | None = None, **metrics) -> None:
    rec = {"ts": time.time(), "epoch": epoch, "step": step, "split": split, **metrics}
    (RUNS_DIR / run_id).mkdir(parents=True, exist_ok=True)
    with open(RUNS_DIR / run_id / "metrics.jsonl", "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
    if split != "train":
        _upsert("runs.json", "runs", {"id": run_id, "epoch": epoch, "last": {split: metrics}})


def update_run(run_id: str, **fields) -> None:
    if fields.get("status") in ("done", "failed"):
        fields["ended"] = time.time()
        log_event(f"run {run_id} {fields['status']}" + (f" best={fields.get('best')}" if fields.get("best") else ""),
                  "success" if fields["status"] == "done" else "error", source=f"run:{run_id}")
    _upsert("runs.json", "runs", {"id": run_id, **fields})


# ---------------------------------------------------------------- results & figures
def set_results(columns: list[str], rows: list[dict], highlight: str = "") -> None:
    with _Lock():
        _write("results.json", {"columns": columns, "rows": rows, "highlight": highlight})


def add_figure(path: str, title: str, group: str = "validation") -> None:
    _upsert("figures.json", "figures", {"id": path, "path": path, "title": title, "group": group, "ts": time.time()})
