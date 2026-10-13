"""Near-duplicate / leakage audit across the dermoscopy datasets (report only, nothing is dropped).

Perceptual hash: 16x16 difference hash (dHash, 256 bits) of the 512-px cache image
(grayscale -> LANCZOS resize to 17x16 -> sign of horizontal gradient).  Pairs with Hamming
distance <= 4 are reported as near-duplicates; counts at <= 10 / <= 20 are given as a sensitivity
analysis.  Exact ISIC-ID overlaps are reported separately.

    python -m cbm.data.overlap
writes data/processed/overlap_report.md, overlap_pairs.csv (all pairs <= 20), overlap_hashes.csv and
overlap_examples.jpg.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from .registry import PROCESSED

DATASETS_AUDITED = ["isic2018_task12", "ph2"]
THR = 4
SENS = [10, 20]


def dhash16(path: str) -> np.ndarray:
    im = Image.open(path).convert("L").resize((17, 16), Image.LANCZOS)
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).ravel().astype(np.uint8)


def _hash_row(args):
    i, p = args
    try:
        return i, dhash16(p)
    except Exception:
        return i, None


def compute_hashes() -> tuple[pd.DataFrame, np.ndarray]:
    rows = []
    for ds in DATASETS_AUDITED:
        m = PROCESSED / ds / "manifest.csv"
        if not m.exists():
            print(f"skip {ds}: no manifest")
            continue
        d = pd.read_csv(m, usecols=["image_id", "split", "image_path", "label_name"])
        d.insert(0, "dataset", ds)
        rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    bits = np.zeros((len(df), 256), np.uint8)
    with mp.get_context("fork").Pool(min(60, os.cpu_count())) as pool:
        for i, h in pool.imap_unordered(_hash_row, list(enumerate(df.image_path)), chunksize=64):
            bits[i] = h
    return df, bits


def all_pairs(bits: np.ndarray, max_d: int) -> pd.DataFrame:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.from_numpy(bits).float().to(dev)
    pop = x.sum(1)
    out_i, out_j, out_d = [], [], []
    B = 4096
    for s in range(0, len(x), B):
        xb = x[s:s + B]
        d = pop[s:s + B, None] + pop[None, :] - 2.0 * (xb @ x.T)
        ii, jj = torch.nonzero(d <= max_d + 0.5, as_tuple=True)
        ii = ii + s
        keep = ii < jj
        out_i.append(ii[keep].cpu()); out_j.append(jj[keep].cpu())
        out_d.append(d[ii[keep] - s, jj[keep]].round().int().cpu())
    return pd.DataFrame({"i": torch.cat(out_i).numpy(), "j": torch.cat(out_j).numpy(),
                         "dist": torch.cat(out_d).numpy()})


def _montage(meta: pd.DataFrame, pairs: pd.DataFrame, path, n=12):
    if pairs.empty:
        return
    pairs = pairs.head(n)
    tile = 160
    gap = 60
    canvas = Image.new("RGB", (tile * 2 + gap, (tile + 22) * len(pairs)), "white")
    dr = ImageDraw.Draw(canvas)
    for k, (_, p) in enumerate(pairs.iterrows()):
        for c, idx in enumerate([p.i, p.j]):
            r = meta.iloc[idx]
            im = Image.open(r.image_path).convert("RGB")
            im.thumbnail((tile, tile))
            canvas.paste(im, (c * (tile + gap), k * (tile + 22) + 20))
            dr.text((c * (tile + gap) + 2, k * (tile + 22) + 2), f"{r.dataset[:8]}:{r.image_id[-7:]}:{r.split}",
                    fill="black")
        dr.text((tile + 15, k * (tile + 22) + 20 + tile // 2), f"d={p.dist}", fill="red")
    canvas.save(path, quality=90)


def run() -> None:
    t0 = time.time()
    meta, bits = compute_hashes()
    out = PROCESSED
    h = meta[["dataset", "image_id", "split"]].copy()
    h["dhash16"] = [np.packbits(b).tobytes().hex() for b in bits]
    h.to_csv(out / "overlap_hashes.csv", index=False)
    pairs = all_pairs(bits, max(SENS))
    for side in ["i", "j"]:
        for c in ["dataset", "image_id", "split", "label_name"]:
            pairs[f"{c}_{side}"] = meta[c].to_numpy()[pairs[side].to_numpy()]
    pairs["same_id"] = pairs.image_id_i == pairs.image_id_j
    pairs["kind"] = np.where(pairs.dataset_i != pairs.dataset_j, "cross_dataset",
                             np.where(pairs.split_i != pairs.split_j, "cross_split", "within_split"))
    pairs["label_agree"] = pairs.label_name_i == pairs.label_name_j
    # order dataset pair names canonically
    a = pairs[["dataset_i", "dataset_j"]].to_numpy()
    pairs["ds_pair"] = [" vs ".join(sorted(x)) for x in a]
    sp = []
    for r in pairs.itertuples():
        x = sorted([f"{r.dataset_i}/{r.split_i}", f"{r.dataset_j}/{r.split_j}"])
        sp.append(" vs ".join(x))
    pairs["split_pair"] = sp
    pairs.sort_values(["kind", "dist"]).to_csv(out / "overlap_pairs.csv", index=False)

    # exact ID overlaps (ISIC IDs are global)
    ids = {ds: set(meta.loc[meta.dataset == ds, "image_id"]) for ds in meta.dataset.unique()}
    id_rows = []
    for i, a_ in enumerate(ids):
        for b_ in list(ids)[i + 1:]:
            inter = ids[a_] & ids[b_]
            if inter:
                sa = meta[(meta.dataset == a_) & meta.image_id.isin(inter)].groupby("split").size().to_dict()
                sb = meta[(meta.dataset == b_) & meta.image_id.isin(inter)].groupby("split").size().to_dict()
                id_rows.append((a_, b_, len(inter), sa, sb))
    # split-level ID overlap table between 2017 and 2018 T1-2
    xt = None
    if {"isic2017", "isic2018_task12"} <= set(ids):
        m17 = meta[meta.dataset == "isic2017"][["image_id", "split"]]
        m18 = meta[meta.dataset == "isic2018_task12"][["image_id", "split"]]
        j = m17.merge(m18, on="image_id", suffixes=("_2017", "_2018"))
        xt = pd.crosstab(j.split_2017, j.split_2018)

    L = ["# Near-duplicate / overlap audit (dermoscopy datasets)", "",
         f"Generated by `cbm.data.overlap` on {time.strftime('%Y-%m-%d %H:%M')}. Nothing was dropped; this is a report only.",
         "",
         "Method: 16x16 difference hash (256 bits) computed on the 512-px cached image (grayscale, LANCZOS to 17x16, "
         "sign of the horizontal gradient). Near-duplicate = Hamming distance <= 4 (of 256). Counts at <= 10 and <= 20 "
         "are a sensitivity analysis (more false positives on dermoscopy, where many images look alike at 16x16). "
         "Global hashes do not detect crops or strong re-colouring, so the counts are lower bounds.", "",
         "Images hashed: " + ", ".join(f"{ds} {int((meta.dataset == ds).sum())}" for ds in meta.dataset.unique()), ""]
    L += ["## 1. Exact ISIC-ID overlaps between datasets", ""]
    if id_rows:
        L += ["| dataset A | dataset B | shared IDs | A splits | B splits |", "|---|---|---|---|---|"]
        for a_, b_, n, sa, sb in id_rows:
            L.append(f"| {a_} | {b_} | {n} | {sa} | {sb} |")
    else:
        L.append("None.")
    if xt is not None and xt.size:
        L += ["", "ISIC 2017 split (rows) x ISIC 2018 Task 1-2 split (columns) for shared IDs:", "",
              xt.to_markdown()]
    L.append("")

    def table(sub: pd.DataFrame, by: str) -> list[str]:
        if sub.empty:
            return ["None.", ""]
        g = sub.groupby(by)
        t = pd.DataFrame({
            f"pairs d<={THR}": g.apply(lambda x: int((x.dist <= THR).sum())),
            f"of which same ID": g.apply(lambda x: int(((x.dist <= THR) & x.same_id).sum())),
            f"of which diff. ID": g.apply(lambda x: int(((x.dist <= THR) & ~x.same_id).sum())),
            f"label disagree (d<={THR})": g.apply(lambda x: int(((x.dist <= THR) & ~x.label_agree).sum())),
            f"unique images (d<={THR})": g.apply(lambda x: len(set(zip(x.dataset_i[x.dist <= THR],
                                                                       x.image_id_i[x.dist <= THR])) |
                                                               set(zip(x.dataset_j[x.dist <= THR],
                                                                       x.image_id_j[x.dist <= THR])))),
            f"pairs d<={SENS[0]}": g.apply(lambda x: int((x.dist <= SENS[0]).sum())),
            f"pairs d<={SENS[1]}": g.apply(lambda x: int((x.dist <= SENS[1]).sum())),
        })
        return [t.to_markdown(), ""]

    cross = pairs[pairs.kind == "cross_dataset"]
    L += ["## 2. Near-duplicates ACROSS datasets", "",
          "Same-ID pairs are the same ISIC image released in two challenges (expected). Different-ID pairs are "
          "the interesting ones (the same lesion photo re-uploaded under a new ID, or a second image of the same lesion).",
          ""]
    L += table(cross, "ds_pair")
    L += [f"Pairs with d<={THR} by dataset/split pair:", ""]
    t2 = cross[cross.dist <= THR].groupby(["split_pair"]).agg(n=("dist", "size"), same_id=("same_id", "sum"))
    L += [t2.to_markdown() if not t2.empty else "None.", ""]
    css = pairs[pairs.kind == "cross_split"]
    L += ["## 3. Near-duplicates across splits WITHIN a dataset (potential leakage)", ""]
    L += table(css, "split_pair")
    wsp = pairs[pairs.kind == "within_split"]
    L += ["## 4. Near-duplicates within the same split (informative)", ""]
    L += table(wsp, "split_pair")

    def examples(sub, title, n=10):
        s = sub[sub.dist <= THR].sort_values("dist").head(n)
        if s.empty:
            s = sub.sort_values("dist").head(min(n, 5))
            title += f" (no pair <= {THR}; closest shown)"
        r = [f"### {title}", ""]
        if s.empty:
            return r + ["None.", ""]
        cols = ["dist", "dataset_i", "image_id_i", "split_i", "label_name_i", "dataset_j", "image_id_j", "split_j",
                "label_name_j"]
        return r + [s[cols].to_markdown(index=False), ""]

    L += ["## 5. Example pairs", ""]
    L += examples(cross[~cross.same_id], "Cross-dataset, different ID")
    L += examples(cross[cross.same_id], "Cross-dataset, same ID", 5)
    for ds in meta.dataset.unique():
        L += examples(css[css.dataset_i == ds], f"{ds}: across splits")
    L += ["Image montage of example pairs: `overlap_examples.jpg` (cross-dataset different-ID first, then "
          "within-dataset cross-split). Full pair list (d<=20): `overlap_pairs.csv`; hashes: `overlap_hashes.csv`.", ""]
    ex = pd.concat([cross[~cross.same_id & (cross.dist <= THR)].sort_values("dist").head(6),
                    css[css.dist <= THR].sort_values("dist").head(6)])
    if ex.empty:
        ex = pd.concat([cross[~cross.same_id].sort_values("dist").head(6), css.sort_values("dist").head(6)])
    _montage(meta, ex, out / "overlap_examples.jpg")
    (out / "overlap_report.md").write_text("\n".join(L))
    print("\n".join(L)[:6000])
    print(f"overlap audit done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    run()
