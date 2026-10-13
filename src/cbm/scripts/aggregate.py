"""Aggregate eval.json of many runs into one table (CSV + markdown), optionally stored in the run registry.
Usage: python src/cbm/scripts/aggregate.py --prefix ebcbm [--out results/ebcbm] [--push]
"""
import argparse, glob, json, os, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))   # src/
from cbm import track as T  # noqa: E402
from cbm.paths import ROOT  # noqa: E402
KEYS = [  # (column name, list of candidate json paths in eval.json, fmt)
 ("test_acc", ["test.task.acc", "task.test.acc", "test_acc"]),
 ("test_bacc", ["test.task.bacc", "task.test.bacc", "test_bacc"]),
 ("test_f1", ["test.task.macro_f1", "task.test.macro_f1", "test.task.f1"]),
 ("test_auroc", ["test.task.auroc_macro", "task.test.auroc_macro", "test.task.auroc"]),
 ("ece", ["test.task.ece", "task.test.ece"]),
 ("concept_auroc", ["test.concepts.mean_auroc", "concepts.test.mean_auroc", "test.concept.mean_auroc"]),
 ("dice", ["test.grounding.mean_dice", "grounding.test.mean_dice", "test.grounding.dice"]),
 ("iou", ["test.grounding.mean_iou", "grounding.test.mean_iou"]),
 ("ois", ["test.leakage.ois", "leakage.ois", "test.leakage.OIS"]),
 ("leak_gap", ["test.leakage.soft_minus_hard_acc", "leakage.soft_minus_hard_acc", "test.leakage.gap"]),
 ("cls", ["test.counterfactual.cls_mean", "counterfactual.cls_mean", "test.counterfactual.CLS"]),
 ("xe", ["test.counterfactual.xe_mean", "counterfactual.xe_mean"]),
 ("tti_auc", ["test.intervention.tti_auc_random", "intervention.tti_auc_random", "test.intervention.auc_random"]),
 ("artefact_flip", ["test.robustness.flip_rate_mean", "robustness.flip_rate_mean", "test.artefacts.flip_rate"]),
 ("ph2_auroc", ["ph2.task.auroc", "ph2.auroc", "ph2.task.auroc_mel"]),
 ("ph2_bacc", ["ph2.task.bacc", "ph2.bacc"]),
]

def get(d, path):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur if isinstance(cur, (int, float)) else None

def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        kk = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, kk + "."))
        elif isinstance(v, (int, float)):
            out[kk] = v
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--prefix", default="ebcbm"); ap.add_argument("--out", default=None)
    ap.add_argument("--push", action="store_true", help="also store the table in experiments/state/results.json")
    a = ap.parse_args()
    rows, flat_rows = [], []
    for ev in sorted(glob.glob(str(ROOT / "experiments/runs" / f"{a.prefix}*" / "eval.json"))):
        run_id = Path(ev).parent.name
        d = json.load(open(ev))
        row = {"run_id": run_id, "method": run_id[len(a.prefix):].rsplit("_s", 1)[0]}
        summ = d.get("summary", {})
        if summ:
            row.update({k: v for k, v in summ.items() if isinstance(v, (int, float))})
        else:
            for col, paths in KEYS:
                v = None
                for p in paths:
                    v = get(d, p)
                    if v is not None: break
                row[col] = v
        rows.append(row)
        fr = flatten(d); fr["run_id"] = run_id; flat_rows.append(fr)
    if not rows:
        print("no eval.json found for prefix", a.prefix); return
    df = pd.DataFrame(rows)
    out = Path(a.out or (ROOT / "results" / f"runs_{a.prefix.strip('_')}"))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(out) + "_table.csv", index=False)
    pd.DataFrame(flat_rows).to_csv(str(out) + "_all_metrics.csv", index=False)
    md = df.to_markdown(index=False, floatfmt=".3f")
    open(str(out) + "_table.md", "w").write(md + "\n")
    print(md)
    if a.push:
        cols = [c for c in df.columns if df[c].notna().any()]
        T.set_results(cols, [{c: (round(r[c], 4) if isinstance(r[c], float) else r[c]) for c in cols} for r in df.to_dict("records")],
                      highlight=f"{len(df)} runs with prefix {a.prefix}")
if __name__ == "__main__":
    main()
