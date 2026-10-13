"""Markdown summary tables (counts per split x class, concept positive rates, cache sizes) for README.md.

    python -m cbm.data.report > data/processed/summary.md
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from .registry import DATASETS, PROCESSED


def _du(path) -> str:
    try:
        return subprocess.run(["du", "-sh", str(path)], capture_output=True, text=True).stdout.split()[0]
    except Exception:
        return "?"


def dataset_section(name: str, with_stats: bool = True) -> str:
    info = DATASETS[name]
    if not os.path.exists(info["manifest"]):
        return f"### {name}\n\nmanifest missing\n"
    df = pd.read_csv(info["manifest"], low_memory=False)
    order = [s for s in info["splits"] + ["excluded"] if s in set(df.split)]
    L = [f"### `{name}`", ""] if with_stats else []
    ct = pd.crosstab(df.label_name, df.split).reindex(columns=order, fill_value=0)
    names = list(info["class_names"]) + [n for n in ct.index if n not in info["class_names"]]
    ct = ct.reindex([n for n in names if n in ct.index])
    ct.index = [f"{n} ({info['class_names'].index(n) if n in info['class_names'] else -1})" for n in ct.index]
    ct.loc["**total**"] = ct.sum()
    ct["all"] = ct.sum(axis=1)
    L += ["Images per split and class (label index in brackets):", "", ct.to_markdown(), ""]
    if "fold" in df:
        f = pd.crosstab(df.label_name, df.fold)
        f.columns = [f"fold {c}" for c in f.columns]
        L += ["CV folds:", "", f.to_markdown(), ""]
    cs = info["concept_names"]
    if cs:
        rows = []
        for c in cs:
            r = {"concept": c}
            for s in order:
                v = df.loc[df.split == s, c]
                r[s] = f"{np.nanmean(v):.3f}" if v.notna().any() else "nan"
            v = df.loc[df.split != "excluded", c]
            r["n_pos (non-excluded)"] = int(np.nansum(v))
            r["n_unknown"] = int(v.isna().sum())
            if f"{c}_area" in df:
                a = df.loc[(df[c] == 1) & (df.split != "excluded"), f"{c}_area"]
                r["median area/lesion (pos)"] = f"{a.median():.3f}"
            rows.append(r)
        L += ["Concept positive rate per split:", "", pd.DataFrame(rows).to_markdown(index=False), ""]
    st = PROCESSED / name / "build_stats.json"
    root = PROCESSED / name
    L += [f"Cache: images_512 {_du(root / 'images_512')} ({len(os.listdir(root / 'images_512'))} files), "
          f"masks_512 {_du(root / 'masks_512')} ({len(os.listdir(root / 'masks_512'))} files).", ""]
    if with_stats and st.exists():
        L += ["<details><summary>build_stats.json</summary>", "", "```json",
              json.dumps(json.load(open(st)), indent=1, default=str)[:6000], "```", "</details>", ""]
    return "\n".join(L)


def main(names=None) -> None:
    for n in names or DATASETS:
        print(dataset_section(n))


if __name__ == "__main__":
    main(sys.argv[1:] or None)
