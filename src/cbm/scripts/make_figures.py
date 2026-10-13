"""Paper figures, Table 1 and the number sheet behind the text (single entry point; CPU only).

    python src/cbm/scripts/make_figures.py --flagship ebcbm_s0 --show_variants \
        [--flagship_variant ebcbm|ebcbm_no_cf|ebcbm_discover|ebcbm_proto|ebcbm_proto_patch] [--example ISIC_x]
        [--proto_run ebcbm_proto_s0]           # adds Fig. 1b (top prototypes with source / nearest training lesions)
        [--only fig1 fig1b fig2 fig3 table1 numbers] [--preset isbi|isbi_spec|large]

Outputs
  figures/paper/fig1_example.{pdf,png}, fig2_completeness_locality_cls.{pdf,png}, fig3_deferral_interventions.{pdf,png}
  (+ fig1.pdf / fig2.pdf / fig3.pdf aliases), results/table1_main.tex (booktabs table*, \\input-able),
  results/table1_main.csv (same numbers, run ids per row), results/final_numbers.json (per-seed values, means, s.d.,
  completeness ladder, sanity checks, intervention curves, oracle ceilings).
Data: experiments/runs/<run_id>/eval.json (+ preds_test.npz and best.pt for Fig. 1, which runs the flagship on one
test image on the CPU).  Runs are grouped into methods by stripping an optional stage prefix (A_, A2_, B_, bl_) and
the seed suffix (_s<k>): ``ebcbm_s0`` and ``ebcbm_s1`` form the method ``ebcbm``.  Seeds are aggregated as
mean +- s.d.  Methods without runs (e.g. the external baselines of the paper, which are not part of this release)
are skipped and listed at the end ("MISSING").
Style: matplotlib, 6.5-7 pt text at 86 mm (column) / 178 mm (full width), colour-blind-safe categorical palette,
identity never by colour alone (axis labels, legends, line styles, hatches).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _here]
sys.path.insert(0, os.path.dirname(os.path.dirname(_here)))   # src/

import numpy as np  # noqa: E402

from cbm.paths import ROOT, RUNS_DIR as RUNS  # noqa: E402

PDF_DIR = ROOT / "figures" / "paper"
PNG_DIR = ROOT / "figures" / "paper"
TAB_DIR = ROOT / "results"
MM = 1 / 25.4
COL_W, FULL_W = 86 * MM, 178 * MM

# validated categorical palette (dataviz reference, light mode; adjacent CVD dE >= 9.1, normal-vision >= 19.6)
C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAGENTA, C_GREEN, C_VIOLET, C_RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#ffffff"

# method key -> (display name, colour, strictly local?)
METHODS = {
    "blackbox": ("Black box (DINOv2)", INK2, False),
    "cbm_joint": ("CBM-joint", C_YELLOW, False),
    "cem": ("CEM", C_ORANGE, False),
    "pcbmh_dino": ("PCBM-h", MUTED, False),
    "lfcbm_clip": ("LF-CBM", MUTED, False),
    "labo": ("LaBo", MUTED, False),
    "explicd": ("Explicd", MUTED, False),
    "sscbm": ("SSCBM", C_VIOLET, False),
    "multitask": ("Multitask seg+cls", MUTED, False),
    "token_pooling": ("EB-CBM (token, approx. local)", C_AQUA, False),
    "ebcbm_no_cf": ("EB-CBM (ours)", C_BLUE, True),
    "ebcbm_discover": ("EB-CBM + discovered regions", C_MAGENTA, True),
    "cbm_joint_inhouse": ("CBM-joint (in-house config)", C_YELLOW, False),
    "cem_inhouse": ("CEM (in-house config)", C_ORANGE, False),
}
TABLE_ORDER = ["blackbox", "cbm_joint", "cem", "pcbmh_dino", "lfcbm_clip", "labo", "explicd", "sscbm", "multitask",
               "token_pooling", "ebcbm_no_cf", "ebcbm_discover"]
CONCEPTS = ["pigment_network", "negative_network", "streaks", "milia_like_cyst", "globules"]
CONCEPT_LABEL = {"pigment_network": "pigment\nnetwork", "negative_network": "negative\nnetwork", "streaks": "streaks",
                 "milia_like_cyst": "milia-like\ncysts", "globules": "globules"}
CONCEPT_COL = [C_BLUE, C_ORANGE, C_AQUA, C_YELLOW, C_MAGENTA]
MISSING: list[str] = []
OUTPUTS: list[str] = []
# figure size presets (mm): isbi = the 4-page paper (default); large = the earlier, roomier layout
# isbi     = the slot sizes in main.tex (178x60, 178x55, 86x60 mm): text >= 6.5 pt at print size, body ends on p. 4
# isbi_spec = 178x65 / 178x55 / 86x70 mm (requested sizes; with matching slots the body spills onto page 5)
PRESETS = {"isbi": {"fig1": (178.0, 60.0), "fig2": (178.0, 55.0), "fig3": (86.0, 60.0), "fs": 6.5},
           "isbi_spec": {"fig1": (178.0, 65.0), "fig2": (178.0, 55.0), "fig3": (86.0, 70.0), "fs": 6.5},
           "large": {"fig1": (178.0, 100.0), "fig2": (178.0, 70.0), "fig3": (86.0, 118.0), "fs": 7.0}}
PRESET_NAME = "isbi"
PRESET = PRESETS[PRESET_NAME]
# v3 family: one is shown as "EB-CBM (ours)" (--flagship_variant); the others are ablation rows (--show_variants)
V3_FAMILY = {"ebcbm_no_cf": ("w/o counterfactual grounding", C_BLUE), "ebcbm_discover": ("+ discovered regions", C_MAGENTA),
             "ebcbm_proto": ("lesion prototypes", C_VIOLET), "ebcbm_proto_patch": ("patch prototypes", C_GREEN),
             "ebcbm": ("counterfactual grounding", C_RED),
             "ebcbm_discover_cf": ("+ discovered regions + CF", MUTED)}
FLAG = "ebcbm"
SHOW_VARIANTS = False
# optional external baseline run ids bl_<cfg>_s<seed> (not part of this release): cfg -> method key
BL_MAP = {"pcbmh_dino": "pcbmh_dino", "lfcbm_clip": "lfcbm_clip", "labo": "labo", "explicd": "explicd",
          "sscbm": "sscbm", "multitask": "multitask", "cem": "cem", "cbm_joint": "cbm_joint"}
# CBM-joint / CEM: the baseline suite's official-recipe runs (bl_cem_s*, bl_cbm_joint_s*) are the table/figure rows when
# the bl_ prefix is given; the in-house configs (cem.yaml / cbm_joint.yaml, A/A2/B runs) become <key>_inhouse (internal
# numbers only, final_numbers.json). --inhouse_cbm restores the in-house rows.
OFFICIAL_BL = {"cem", "cbm_joint"}
USE_OFFICIAL_BL = True
# ablation rows under "EB-CBM (ours)" with --show_variants: flagship -> [(variant key, row label)]
ABLATIONS = {"ebcbm": [("ebcbm_no_cf", "w/o counterfactual grounding"),
                              ("ebcbm_discover", "+ discovered regions")]}
FIG_VARIANTS = False
TABLE_FONT = "scriptsize"     # scriptsize keeps the body on 4 pages with the two ablation rows (footnotesize: 5)
CITE = {"cbm_joint": "koh2020cbm", "cem": "espinosazarlenga2022cem", "pcbmh_dino": "yuksekgonul2023pcbm",
        "lfcbm_clip": "oikarinen2023lfcbm", "labo": "yang2023labo", "explicd": "gao2024explicd", "sscbm": "hu2025sscbm"}


def set_flagship(key: str, show_variants: bool):
    """Register the chosen v3-family method as 'EB-CBM (ours)' and the others as ablation variants."""
    global FLAG, SHOW_VARIANTS
    FLAG, SHOW_VARIANTS = key, show_variants
    for k, (lab, col) in V3_FAMILY.items():
        METHODS[k] = ("EB-CBM (ours)", C_BLUE, True) if k == key else (f"EB-CBM, {lab}", col, True)


def variants(R) -> list[str]:
    """Ablation rows shown with --show_variants (ABLATIONS[flagship] if defined, else every other v3 variant)."""
    if not SHOW_VARIANTS:
        return []
    if FLAG in ABLATIONS:
        return [k for k, _ in ABLATIONS[FLAG] if R.get(k)]
    return [k for k in V3_FAMILY if k != FLAG and R.get(k)]


def variant_label(key: str) -> str:
    return dict(ABLATIONS.get(FLAG, [])).get(key, V3_FAMILY.get(key, (key,))[0])


# ============================================================================ run discovery / aggregation
def method_key(run_id: str) -> tuple[str, str, int] | None:
    """``[<stage>_]<method>_s<seed>`` -> (stage or "", method, seed); stage in {A, A2, B, bl} (optional)."""
    m = re.match(r"^(?:(A2|A|B|bl)_)?(.+)_s(\d+)$", run_id)
    if not m:
        return None
    return m.group(1) or "", m.group(2), int(m.group(3))


def discover_runs(prefixes: list[str]) -> dict[str, list[str]]:
    """method key -> run ids (one per seed; A2_ replaces A_ for the same seed). Run ids without a stage prefix
    (``<method>_s<seed>``) are always included; optional prefixes A,B,bl ("_" optional) add staged runs; bl_ runs are
    mapped to method keys with BL_MAP (unmapped configs are ignored)."""
    prefixes = [p.rstrip("_") for p in prefixes]
    allowed = set(prefixes) | {""} | ({"A2"} if "A" in prefixes else set())   # "" = no stage prefix
    best: dict[tuple[str, int], tuple[int, str]] = {}
    rank = {"": 0, "A": 0, "B": 1, "A2": 2, "bl": 0}
    for d in sorted(RUNS.iterdir()):
        mk = method_key(d.name)
        if not mk or mk[0] not in allowed or not (d / "eval.json").exists():
            continue
        stage, key, seed = mk
        if stage == "bl":
            if key not in BL_MAP or (key in OFFICIAL_BL and not USE_OFFICIAL_BL):
                continue
            key = BL_MAP[key]
        elif key in OFFICIAL_BL and USE_OFFICIAL_BL and "bl" in allowed:
            key = f"{key}_inhouse"
        cur = best.get((key, seed))
        if cur is None or rank[stage] > cur[0]:
            best[(key, seed)] = (rank[stage], d.name)
    out = defaultdict(list)
    for (key, seed), (_, rid) in sorted(best.items()):
        out[key].append(rid)
    return dict(out)


_EV: dict[str, dict] = {}


def ev(rid: str) -> dict:
    if rid not in _EV:
        _EV[rid] = json.load(open(RUNS / rid / "eval.json"))
    return _EV[rid]


def dig(d, path: str):
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def agg(runs: list[str], path: str, what: str = "") -> tuple[float, float, int]:
    vals = [dig(ev(r), path) for r in runs]
    vals = [float(v) for v in vals if isinstance(v, (int, float)) and np.isfinite(v)]
    if not vals:
        if what:
            MISSING.append(f"{what}: {path} not found in {runs}")
        return float("nan"), float("nan"), 0
    return float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, len(vals)


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": PRESET["fs"], "axes.titlesize": PRESET["fs"] + 0.5,
        "axes.labelsize": PRESET["fs"], "xtick.labelsize": PRESET["fs"], "ytick.labelsize": PRESET["fs"],
        "legend.fontsize": PRESET["fs"], "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "legend.frameon": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "pdf.fonttype": 42, "ps.fonttype": 42, "lines.linewidth": 1.4, "patch.linewidth": 0,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5})
    return plt


def save(fig, name: str, title: str):
    from cbm import track as T
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    pdf, png = PDF_DIR / f"{name}.pdf", PNG_DIR / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=300)
    OUTPUTS.extend([str(pdf), str(png)])
    alias = {"fig1_example": "fig1", "fig2_completeness_locality_cls": "fig2", "fig3_deferral_interventions": "fig3",
             "fig1b_prototypes": "fig1b"}.get(name)
    if alias:                                   # names included by main.tex
        import shutil
        shutil.copyfile(pdf, PDF_DIR / f"{alias}.pdf")
        OUTPUTS.append(str(PDF_DIR / f"{alias}.pdf"))
    try:
        T.add_figure(str(png), title, group="paper")
    except Exception as e:  # registry update is best effort
        print("add_figure failed:", e)


def panel_label(ax, s):
    ax.text(-0.02, 1.02, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right", color=INK)


# ============================================================================ Fig. 1
def _pick_example(rid: str) -> str:
    d = np.load(RUNS / rid / "preds_test.npz", allow_pickle=True)
    y, p = d["y_true"], d["y_prob"]
    K = len(CONCEPTS)
    pred_present = (d["mask_pred"][:, :K] >= 128).reshape(len(y), K, -1).mean(-1) >= 0.002
    gt = np.nan_to_num(d["c_true"]) > 0.5
    ok = (y == 1) & (p.argmax(1) == 1) & (gt.sum(1) >= 2)
    score = (pred_present & gt).sum(1) - 0.5 * (pred_present & ~gt).sum(1) + p[:, 1]
    score[~ok] = -1e9
    return str(d["image_id"][int(np.argmax(score))])


def fig1(flagship: str, example: str | None = None):
    import torch
    import torch.nn.functional as F
    from torchvision.ops import roi_align
    from cbm.datamod import DermDataset, collate, dataset_kwargs, denormalize
    from cbm.models import load_model
    from cbm.models.heads import square_boxes
    plt = style()
    from matplotlib.patches import FancyBboxPatch, Patch
    import matplotlib.gridspec as gridspec
    run_dir = RUNS / flagship
    model, cfg, _ = load_model(run_dir, "cpu")
    model.eval()
    iid = example or _pick_example(flagship)
    ds = DermDataset(cfg["data"]["dataset"], "test", **dataset_kwargs(cfg))
    i = ds.ids.index(iid)
    b = collate([ds[i]])
    x = b["image"]
    with torch.no_grad():
        out = model(x)
    K = model.K
    masks = out["masks"][0].numpy()                                         # (P,128,128)
    gt = b["masks"][0].numpy()                                              # (K+1,128,128)
    p = torch.sigmoid(out["concept_logits"][0]).numpy()
    prob = torch.softmax(out["logits"][0], -1).numpy()
    hin = out["extra"]["head_inputs"][0].numpy()
    names = model.head_input_names
    W = model.diag.weight.detach().numpy()
    bias = model.diag.bias.detach().numpy()
    yhat = int(prob.argmax())
    classes = list(ds.class_names)
    img = denormalize(x[0]).permute(1, 2, 0).numpy()
    H = img.shape[0]
    # crops exactly as the local encoder sees them (concept planes) + the lesion crop of the backbone pass
    Mb = (out["masks"][:, :K].float() > 0.5).float()
    inp, sel, empty = model.concept_head.crops(x, Mb)
    crops = []
    if inp is not None:
        for j, s in enumerate(sel.tolist()):
            c = s % K
            keep = inp[j, 3].numpy() > 0.5 if inp.shape[1] > 3 else np.ones(inp.shape[-2:], bool)
            im = denormalize(inp[j:j + 1, :3])[0].permute(1, 2, 0).numpy()
            im[~keep] = 1.0
            crops.append((CONCEPTS[c].replace("_", " "), im, CONCEPT_COL[c]))
    if getattr(model, "lesion_source", None) in ("crop_backbone", "crop_backbone_proto"):
        Lb = (out["masks"][:, K].float() > 0.5).float()
        if Lb.mean() >= 0.002:
            bx = square_boxes(Lb, (H, H), model.lesion_margin)
            S = 224
            xc = roi_align(x.float(), torch.cat([torch.zeros(1, 1), bx], 1), (S, S), 1.0, 0, True)
            mc = roi_align(Lb[:, None], torch.cat([torch.zeros(1, 1), bx], 1), (S, S), Lb.shape[-1] / H, 0, True)
            im = denormalize(xc)[0].permute(1, 2, 0).numpy()
            im[mc[0, 0].numpy() < 0.999] = 1.0
            crops.append(("lesion (backbone)", im, INK2))
    # contributions of the input groups to the predicted class (centred weights: softmax-invariant)
    Wc = W[yhat] - W.mean(0)
    contrib = Wc * hin
    groups = [n.split(":")[0] for n in names]
    gnames = [g for g in ["concept", "area", "appearance", "report", "lesion", "part"] if g in groups]
    gsum = {g: float(sum(c for c, gg in zip(contrib, groups) if gg == g)) for g in gnames}
    gsum["bias"] = float(bias[yhat] - bias.mean())
    rep = out["extra"].get("report")
    rep_names = model.report_names
    if PRESET_NAME != "large":   # isbi / isbi_spec
        _fig1_compact(plt, iid, img, H, gt, masks, crops, p, rep, rep_names, prob, classes, yhat, gsum, K, flagship)
        return iid
    # ---------------------------------------------------------------- layout (mm coordinates, 178 x 104 mm)
    FW, FH = 178.0, 100.0
    fig = plt.figure(figsize=(FW * MM, FH * MM))

    def axmm(x, y, w, h):
        return fig.add_axes([x / FW, y / FH, w / FW, h / FH])

    def show(ax, im, title, fs=7):
        ax.imshow(im)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, fontsize=fs, color=INK, pad=2)

    import cv2

    def contour(ax, m, col, lw=1.0, ls="-"):
        mm = (cv2.resize(m.astype(np.float32), (H, H)) >= 0.5).astype(np.uint8)
        cs, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cs:
            c = c[:, 0]
            ax.plot(np.r_[c[:, 0], c[0, 0]], np.r_[c[:, 1], c[0, 1]], color=col, lw=lw, ls=ls)

    def label(x, y, t):
        fig.text(x / FW, y / FH, t, fontsize=8, fontweight="bold", color=INK, va="top", ha="left")
    S_IMG, Y_IMG = 38.0, 54.0
    a0 = axmm(6, Y_IMG, S_IMG, S_IMG); show(a0, img, f"input ({iid})")
    a1 = axmm(48, Y_IMG, S_IMG, S_IMG); show(a1, img, "expert masks")
    a2 = axmm(90, Y_IMG, S_IMG, S_IMG); show(a2, img, "predicted masks")
    label(1, 98, "a")
    for k in range(K):
        contour(a1, gt[k] >= 0.5, CONCEPT_COL[k], 1.0)
        contour(a2, masks[k] >= 0.5, CONCEPT_COL[k], 1.0)
    contour(a1, gt[K] >= 0.5, SURFACE, 0.9, "--")
    contour(a2, masks[K] >= 0.5, SURFACE, 0.9, "--")
    handles = [Patch(color=CONCEPT_COL[k], label=CONCEPTS[k].replace("_", " ")) for k in range(K)]
    handles.append(plt.Line2D([], [], color=MUTED, ls="--", lw=0.9, label="lesion (white dashed)"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(6 / FW, 45.5 / FH), ncol=6, fontsize=6.5,
               handlelength=1.0, columnspacing=1.0, borderaxespad=0)
    # crops (up to 4 in a 2x2 block)
    label(134, 98, "b")
    fig.text(156 / FW, 98 / FH, "what the local encoders see", fontsize=7, color=INK, ha="center", va="top")
    cs_ = 19.0
    pos = [(137, 72), (157.5, 72), (137, 48.5), (157.5, 48.5)]
    for j in range(4):
        ax = axmm(pos[j][0], pos[j][1], cs_, cs_)
        if j < len(crops):
            show(ax, crops[j][1], crops[j][0], fs=6.5)
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color(crops[j][2]); sp.set_linewidth(1.2)
        else:
            ax.axis("off")
    if len(crops) > 4:
        MISSING.append(f"fig1: {len(crops)} crops, only 4 shown")
    # ---------------------------------------------------------------- schematic
    ax = axmm(4, 3, 58, 38); ax.set_xlim(0, 10.4); ax.set_ylim(0, 10); ax.axis("off")
    label(1, 43, "c")

    def box(x0, y0, w, h, txt, fc):
        ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.25", fc=fc, ec="none"))
        ax.text(x0 + w / 2, y0 + h / 2, txt, ha="center", va="center", fontsize=6.3, color=INK, linespacing=1.15)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.7, color=INK2,
                                                                       shrinkA=0, shrinkB=0, mutation_scale=6))
    box(0.0, 7.3, 1.9, 1.9, "image", "#f0efec")
    box(2.5, 7.3, 3.6, 1.9, "global DINOv2\n+ mask decoder", "#cde2fb")
    box(6.8, 7.3, 3.6, 1.9, "K+1 masks", "#cde2fb")
    box(6.3, 3.6, 4.1, 2.6, "shared local\nencoder on\nmasked crops", "#d6efe5")
    box(1.6, 3.6, 4.1, 2.6, "report: areas,\nshape, colours\n(from masks)", "#fbe3d6")
    box(3.0, 0.2, 4.4, 2.2, "linear head\n-> diagnosis", "#e8e6f4")
    arrow(1.9, 8.25, 2.5, 8.25); arrow(6.1, 8.25, 6.8, 8.25)
    arrow(8.6, 7.3, 8.4, 6.2); arrow(7.4, 7.3, 4.6, 6.2)
    arrow(8.0, 3.6, 6.4, 2.4); arrow(3.6, 3.6, 4.4, 2.4)
    ax.text(5.2, 9.75, "only the masks leave the global pass", fontsize=6.3, color=INK2, ha="center")
    # ---------------------------------------------------------------- table: concepts + report
    label(66, 43, "d")
    ax = axmm(68, 3, 52, 38); ax.axis("off")
    rows = [[CONCEPTS[k].replace("_", " "), "yes" if gt[k].max() >= 0.5 else "no", f"{p[k]:.2f}"] for k in range(K)]
    if rep is not None:
        r = rep[0].numpy()
        pick = ["lesion:lesion_asymmetry", "lesion:border_irregularity", "lesion:colour_count", "lesion:lesion_area"]
        for nm in pick:
            if nm in rep_names:
                rows.append([nm.split(":")[1].replace("_", " "), "", f"{r[rep_names.index(nm)]:.2f}"])
    tb = ax.table(cellText=rows, colLabels=["concept / descriptor", "expert", "model"], loc="upper left",
                  cellLoc="left", colLoc="left", colWidths=[0.6, 0.2, 0.2], bbox=[0, 0, 1, 1])
    tb.auto_set_font_size(False); tb.set_fontsize(6.5)
    for (rr, cc), cell in tb.get_celld().items():
        cell.set_edgecolor(GRID); cell.set_linewidth(0.4)
        if rr == 0:
            cell.set_text_props(color=INK, fontweight="bold")
    # ---------------------------------------------------------------- diagnosis + contributions
    label(124, 43, "e")
    ax = axmm(147, 31, 28, 10)
    yy = np.arange(len(classes))
    ax.barh(yy, prob, height=0.6, color=[C_BLUE if j == yhat else MUTED for j in range(len(classes))])
    ax.set_yticks(yy); ax.set_yticklabels([c.replace("_", " ").replace("seborrheic keratosis", "seb. keratosis")
                                           for c in classes], fontsize=6.5)
    ax.invert_yaxis(); ax.set_xlim(0, 1.25); ax.set_xticks([0, 0.5, 1]); ax.grid(axis="y", visible=False)
    ax.set_xlabel("P(class)", labelpad=1)
    for j, v in enumerate(prob):
        ax.text(v + 0.03, j, f"{v:.2f}", va="center", fontsize=6.3, color=INK)
    ax = axmm(147, 8.5, 26, 14.5)
    ks = list(gsum)
    vals = [gsum[k] for k in ks]
    yy = np.arange(len(ks))
    ax.barh(yy, vals, height=0.6, color=[C_BLUE if v >= 0 else C_RED for v in vals])
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_yticks(yy); ax.set_yticklabels(ks, fontsize=6.5); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    lo, hi = min(0.0, min(vals)), max(0.0, max(vals))
    ax.set_xlim(lo - 0.05 * (hi - lo + 1e-6), hi + 0.35 * (hi - lo + 1e-6))
    for j, v in enumerate(vals):
        ax.text(max(v, 0) + 0.04 * (hi - lo + 1e-6), j, f"{v:+.2f}", va="center", fontsize=6, color=INK)
    ax.set_xlabel(f"logit contribution ({classes[yhat].replace('_', ' ')})", labelpad=1)
    save(fig, "fig1_example", f"Fig. 1: flagship example {iid} ({flagship})")
    plt.close(fig)
    return iid


CROP_SHORT = {"pigment network": "pigm. netw.", "negative network": "neg. netw.", "streaks": "streaks",
              "milia like cyst": "milia cysts", "globules": "globules", "lesion (backbone)": "lesion"}


def _fig1_compact(plt, iid, img, H, gt, masks, crops, p, rep, rep_names, prob, classes, yhat, gsum, K, flagship):
    """Fig. 1 for the 4-page paper (178 x 65 mm, text >= 6.5 pt): (a) input / expert / predicted masks, (b) the four
    crops the local encoders see, (c) concept + descriptor table, (d) class probabilities and per-group logit
    contributions. The architecture schematic is left to the text."""
    import cv2
    from matplotlib.patches import Patch
    FW, FH = PRESET["fig1"]
    FS = PRESET["fs"]
    fig = plt.figure(figsize=(FW * MM, FH * MM))

    def axmm(x0, y0, w, h):
        return fig.add_axes([x0 / FW, y0 / FH, w / FW, h / FH])

    def label(x0, y0, t):
        fig.text(x0 / FW, y0 / FH, t, fontsize=FS + 1.5, fontweight="bold", color=INK, va="top", ha="left")

    def show(ax, im, title):
        ax.imshow(im); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, fontsize=FS, color=INK, pad=1.5)

    def contour(ax, m, col, lw=0.9, ls="-"):
        mm = (cv2.resize(m.astype(np.float32), (H, H)) >= 0.5).astype(np.uint8)
        cs, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cs:
            c = c[:, 0]
            ax.plot(np.r_[c[:, 0], c[0, 0]], np.r_[c[:, 1], c[0, 1]], color=col, lw=lw, ls=ls)
    S_IMG = min(30.5, FH - 31.5)                     # 30.5 mm at 65 mm height, 28.5 mm at 60 mm
    Y_IMG = FH - S_IMG - 4.5
    xs = [3.0, 3.0 + S_IMG + 2.0, 3.0 + 2 * (S_IMG + 2.0)]
    TB_H = Y_IMG - 11.5                               # table block height
    a0 = axmm(xs[0], Y_IMG, S_IMG, S_IMG); show(a0, img, "input")
    a1 = axmm(xs[1], Y_IMG, S_IMG, S_IMG); show(a1, img, "expert masks")
    a2 = axmm(xs[2], Y_IMG, S_IMG, S_IMG); show(a2, img, "predicted masks")
    for k in range(K):
        contour(a1, gt[k] >= 0.5, CONCEPT_COL[k])
        contour(a2, masks[k] >= 0.5, CONCEPT_COL[k])
    contour(a1, gt[K] >= 0.5, SURFACE, 0.8, "--")
    contour(a2, masks[K] >= 0.5, SURFACE, 0.8, "--")
    label(0.3, FH - 0.3, "a")
    short = ["pigment netw.", "negative netw.", "streaks", "milia-like cysts", "globules"]
    handles = [Patch(color=CONCEPT_COL[k], label=short[k]) for k in range(K)]
    handles.append(plt.Line2D([], [], color=MUTED, ls="--", lw=0.8, label="lesion"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(2.0 / FW, (Y_IMG - 8.0) / FH), ncol=3, fontsize=FS,
               handlelength=1.0, handleheight=0.7, columnspacing=0.9, labelspacing=0.25, borderaxespad=0)
    # (b) crops
    label(101.5, FH - 0.3, "b")
    cs_ = 16.4
    for j in range(4):
        ax = axmm(104.5 + j * 18.0, Y_IMG + S_IMG - cs_, cs_, cs_)
        if j < len(crops):
            show(ax, crops[j][1], CROP_SHORT.get(crops[j][0], crops[j][0]))
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_color(crops[j][2]); sp.set_linewidth(1.0)
        else:
            ax.axis("off")
    if len(crops) > 4:
        MISSING.append(f"fig1: {len(crops)} crops, only 4 shown")
    # (c) table: 5 concepts (expert / model) + 3 report descriptors
    label(0.3, TB_H + 2.5, "c")
    ax = axmm(4.0, 1.0, 58.0, TB_H); ax.axis("off")
    rows = [[CONCEPTS[k].replace("_", " ").replace("milia like cyst", "milia-like cysts"),
             "yes" if gt[k].max() >= 0.5 else "no", f"{p[k]:.2f}"] for k in range(K)]
    tb = ax.table(cellText=rows, colLabels=["structure", "expert", "model"], cellLoc="left", colLoc="left",
                  colWidths=[0.56, 0.22, 0.22], bbox=[0, 0, 1, 1])
    tb.auto_set_font_size(False); tb.set_fontsize(FS)
    ax2 = axmm(64.0, 1.0 + TB_H * 2 / 6, 34.5, TB_H * 4 / 6); ax2.axis("off")     # same row height as (c) left
    drows = []
    if rep is not None:
        r = rep[0].numpy()
        for nm, lab in [("lesion:lesion_asymmetry", "asymmetry"), ("lesion:border_irregularity", "border irreg."),
                        ("lesion:colour_count", "colours")]:
            if nm in rep_names:
                drows.append([lab, f"{r[rep_names.index(nm)]:.2f}"])
    else:
        MISSING.append("fig1: no report descriptors in the flagship model")
    tb2 = ax2.table(cellText=drows or [["--", "--"]], colLabels=["descriptor", "value"], cellLoc="left",
                    colLoc="left", colWidths=[0.66, 0.34], bbox=[0, 0, 1, 1])
    tb2.auto_set_font_size(False); tb2.set_fontsize(FS)
    for t in (tb, tb2):
        for (rr, cc), cell in t.get_celld().items():
            cell.set_edgecolor(GRID); cell.set_linewidth(0.4)
            cell.PAD = 0.04
            if rr == 0:
                cell.set_text_props(color=INK, fontweight="bold")
    # (d) per-group logit contributions (predicted class and its probability in the title)
    top_d = Y_IMG + S_IMG - cs_ - 1.5
    label(101.5, top_d, "d")
    ax = axmm(124.0, 7.0, 45.0, top_d - 7.0 - 5.0)
    ks = list(gsum)
    vals = [gsum[k] for k in ks]
    yy = np.arange(len(ks))
    ax.barh(yy, vals, height=0.62, color=[C_BLUE if v >= 0 else C_RED for v in vals])
    ax.axvline(0, color=INK2, lw=0.6)
    ax.set_yticks(yy); ax.set_yticklabels(ks); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
    lo, hi = min(0.0, min(vals)), max(0.0, max(vals))
    ax.set_xlim(lo - 0.05 * (hi - lo + 1e-6), hi + 0.42 * (hi - lo + 1e-6))
    for j, v in enumerate(vals):
        ax.text(max(v, 0) + 0.04 * (hi - lo + 1e-6), j, f"{v:+.2f}", va="center", fontsize=FS, color=INK)
    cname = classes[yhat].replace("_", " ")
    ax.set_xlabel(f"contribution to the {cname} logit", labelpad=0.5)
    ax.set_title(f"diagnosis: {cname}, p = {prob[yhat]:.2f}", loc="left", fontsize=FS, pad=2)
    save(fig, "fig1_example", f"Fig. 1: flagship example {iid} ({flagship})")
    plt.close(fig)


# ============================================================================ Fig. 1b (prototype runs)
def _lesion_crop_img(model, x, out=None, S=224):
    """Masked square lesion crop (as the second backbone pass sees it) for display; returns (img, box, crop_size)."""
    import torch
    from torchvision.ops import roi_align
    from cbm.datamod import denormalize
    from cbm.models.heads import square_boxes
    K = model.K
    if out is None:
        with torch.no_grad():
            out = model(x)
    H = x.shape[-1]
    Lb = (out["masks"][:, K].float() > 0.5).float()
    if Lb.mean() < 0.002:
        return np.ones((S, S, 3)), None
    bx = square_boxes(Lb, (H, H), model.lesion_margin)
    xc = roi_align(x.float(), torch.cat([torch.zeros(1, 1), bx], 1), (S, S), 1.0, 0, True)
    mc = roi_align(Lb[:, None], torch.cat([torch.zeros(1, 1), bx], 1), (S, S), Lb.shape[-1] / H, 0, True)
    im = denormalize(xc)[0].permute(1, 2, 0).numpy()
    im[mc[0, 0].numpy() < 0.999] = 1.0
    return im, bx


def fig1_proto(proto_run: str, example: str | None = None, top_k: int = 3, n_near: int = 5):
    """Fig. 1b for a crop_backbone_proto run: the example's lesion crop, the top-k prototypes by contribution to the
    predicted class (centred weight x similarity), each with its source training lesion (projected prototype) and the
    5 nearest training lesions (prototypes_final.json)."""
    import torch
    from cbm.datamod import DermDataset, collate, dataset_kwargs
    from cbm.models import load_model
    plt = style()
    from matplotlib.patches import Rectangle
    run_dir = RUNS / proto_run
    model, cfg, _ = load_model(run_dir, "cpu")
    model.eval()
    if getattr(model, "lesion_source", "") != "crop_backbone_proto":
        MISSING.append(f"fig1b: {proto_run} is not a prototype run")
        return None
    iid = example or _pick_example(proto_run)
    kw = dataset_kwargs(cfg)
    te = DermDataset(cfg["data"]["dataset"], "test", **kw)
    tr = DermDataset(cfg["data"]["dataset"], "train", **kw)
    x = collate([te[te.ids.index(iid)]])["image"]
    with torch.no_grad():
        out = model(x)
    prob = torch.softmax(out["logits"][0], -1).numpy()
    yhat = int(prob.argmax())
    names = model.head_input_names
    hin = out["extra"]["head_inputs"][0].numpy()
    W = model.diag.weight.detach().numpy()
    Wc = W[yhat] - W.mean(0)
    pidx = [i for i, n in enumerate(names) if n.startswith("lesion:proto")]
    contrib = np.array([Wc[i] * hin[i] for i in pidx])
    sims = out["extra"]["proto_sim"][0].numpy()
    order = np.argsort(-contrib)[:top_k]
    pj = run_dir / "prototypes_final.json"
    pinfo = json.load(open(pj)) if pj.exists() else (json.load(open(run_dir / "prototypes.json"))
                                                    if (run_dir / "prototypes.json").exists() else None)
    if pinfo is None:
        MISSING.append(f"fig1b: no prototypes*.json in {proto_run}")
    src = [f"ISIC_{int(v):07d}" if int(v) >= 0 else "" for v in model.proto_src.tolist()]
    classes = list(te.class_names)
    patch_mode = model.proto_pool == "max_patch"

    def crop_of(image_id, ds):
        xx = collate([ds[ds.ids.index(image_id)]])["image"]
        return _lesion_crop_img(model, xx)[0]
    FW, FH = 178.0, 64.0
    fig = plt.figure(figsize=(FW * MM, FH * MM))

    def axmm(x0, y0, w, h):
        return fig.add_axes([x0 / FW, y0 / FH, w / FW, h / FH])

    def show(ax, im, title="", fs=6.5, col=None):
        ax.imshow(im); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(col is not None)
            if col:
                sp.set_color(col); sp.set_linewidth(1.0)
        if title:
            ax.set_title(title, fontsize=fs, color=INK, pad=2)
    ax = axmm(4, 14, 36, 36)
    im, _ = _lesion_crop_img(model, x, out)
    show(ax, im, f"{iid}: lesion crop\nP({classes[yhat].replace('_', ' ')}) = {prob[yhat]:.2f}")
    fig.text(1 / FW, 62 / FH, "a", fontsize=8, fontweight="bold", va="top")
    fig.text(44 / FW, 62 / FH, "b", fontsize=8, fontweight="bold", va="top")
    bw = (FW - 48) / top_k
    for r, j in enumerate(order):
        x0 = 46 + r * bw
        a = axmm(x0, 22, 28, 28)
        sid = src[j] if src[j] else (pinfo["prototypes"][j]["src_image_id"] if pinfo else "")
        if sid and sid in tr.ids:
            show(a, crop_of(sid, tr), f"prototype {j}: sim {sims[j]:.2f}\ncontribution {contrib[j]:+.2f}", col=C_BLUE)
            if patch_mode and int(model.proto_src_patch[j]) >= 0:
                pi = int(model.proto_src_patch[j]); g = 32; cs = 224 / g
                a.add_patch(Rectangle(((pi % g) * cs, (pi // g) * cs), cs, cs, fill=False, ec=C_RED, lw=1.0))
            a.text(0.0, -0.06, f"source: {sid}", transform=a.transAxes, fontsize=5.8, color=INK2, va="top")
        else:
            a.axis("off"); MISSING.append(f"fig1b: source lesion of prototype {j} not found ({sid})")
        if pinfo:
            near = [d["image_id"] for d in pinfo["prototypes"][j]["nearest_train"] if d["image_id"] != sid][:n_near - 1]
            tw = min(bw - 2, 40) / max(len(near), 1)
            for t, nid in enumerate(near):
                at = axmm(x0 + t * tw, 3, tw - 0.8, tw - 0.8)
                if nid in tr.ids:
                    show(at, crop_of(nid, tr))
                else:
                    at.axis("off")
            if r == 0:
                fig.text(x0 / FW, 1 / FH + (tw + 2.5) / FH, "nearest training lesions", fontsize=6, color=INK2)
    save(fig, "fig1b_prototypes", f"Fig. 1b: top prototypes for {iid} ({proto_run})")
    plt.close(fig)
    return iid


# ============================================================================ Fig. 2
LADDER_LABEL = {"concept": "concepts", "area": "+ areas", "report": "+ report", "appearance": "+ appearance",
                "lesion": "+ lesion", "part": "+ parts"}


def _ladder(runs: list[str]) -> dict[tuple, list[float]]:
    out = defaultdict(list)
    for r in runs:
        lad = ev(r).get("ladder") or {}
        for st in lad.get("steps", []):
            out[tuple(st["groups"])].append(st["refit"])
    return out


def fig2(R: dict[str, list[str]], flag_key: str):
    plt = style()
    FW, FH = PRESET["fig2"]
    FS = PRESET["fs"]
    big = PRESET_NAME == "large"
    fig = plt.figure(figsize=(FW * MM, FH * MM))
    # geometry (mm): axes bottom / height, panel x positions and widths (fig 2 is 55 mm in both isbi presets)
    yb, ah = (22.0, 40.0) if big else (19.0, 29.0)

    def axmm(x, y, w, h):
        return fig.add_axes([x / FW, y / FH, w / FW, h / FH])

    def label(x, y, t):
        fig.text(x / FW, y / FH, t, fontsize=FS + 1.5, fontweight="bold", color=INK, va="top", ha="left")
    # (a) completeness ladder ------------------------------------------------------------------------------
    ax = axmm(11.5, yb, 42, ah)
    label(0.5, FH - 0.5, "a")
    lad = _ladder(R.get(flag_key, []))
    steps = sorted(lad, key=len)
    if not steps:
        MISSING.append(f"fig2a: no ladder in {R.get(flag_key)}")
    disc_key = "ebcbm_discover" if flag_key != "ebcbm_discover" else None
    disc = _ladder(R.get(disc_key, [])) if disc_key else {}
    part_step = [s_ for s_ in disc if "part" in s_]
    short_lab = {"concept": "concepts", "area": "+areas", "report": "+report", "appearance": "+appear.",
                 "lesion": "+lesion", "part": "+parts"}
    labels, means, sds, cols = [], [], [], []
    for st in steps:
        labels.append((LADDER_LABEL if big else short_lab).get(st[-1], st[-1])); v = lad[st]
        means.append(np.mean(v)); sds.append(np.std(v, ddof=1) if len(v) > 1 else 0); cols.append(C_BLUE)
    if part_step:
        v = disc[part_step[0]]
        labels.append("+ parts (disc.)" if big else "+parts*"); means.append(np.mean(v))
        sds.append(np.std(v, ddof=1) if len(v) > 1 else 0)
        cols.append(C_MAGENTA)
    elif disc_key:
        MISSING.append("fig2a: no '+part' ladder step (discover run)")
    xx = np.arange(len(labels))
    ax.bar(xx, means, width=0.6, color=cols, yerr=sds if any(sds) else None, error_kw=dict(lw=0.6, capsize=1.5))
    for j, m in enumerate(means):
        ax.text(j, m + 0.008, f"{m:.2f}"[1:] if not big else f"{m:.2f}", ha="center", va="bottom", fontsize=FS,
                color=INK)
    bb = agg(R.get("blackbox", []), "summary.test_bacc", "fig2a black-box BACC")[0]
    oc = agg(R.get(flag_key, []), "sanity.oracle_concepts.test_bacc", "fig2a oracle concepts")[0]
    oca = agg(R.get(flag_key, []), "sanity.oracle_concepts_areas.test_bacc", "fig2a oracle concepts+areas")[0]
    hl = []
    for val, lab, ls in [(bb, "black box", "-"), (oca, "oracle conc.+areas", ":"), (oc, "oracle concepts", "--")]:
        if np.isfinite(val):
            hl.append(ax.axhline(val, color=INK2, lw=0.8, ls=ls, label=f"{lab} ({val:.2f})"))
    ax.legend(handles=hl, loc="upper left", fontsize=FS, handlelength=1.8, borderaxespad=0.1, labelspacing=0.2)
    ax.set_xticks(xx); ax.set_xticklabels(labels, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylim(0.3, 0.98 if not big else 0.86); ax.set_ylabel("test BACC (refit)")
    ax.grid(axis="x", visible=False)
    ax.set_title("completeness ladder", loc="left", pad=2)
    # (b) locality ------------------------------------------------------------------------------------------
    ax = axmm(67.0, yb, 52, ah)
    label(56.5, FH - 0.5, "b")
    mk = [m for m in ["cbm_joint", "cem", "sscbm", "token_pooling", flag_key] + (variants(R) if FIG_VARIANTS else [])
          if R.get(m)]
    cd, tv = [], []
    for m in mk:
        runs = R[m]
        if dig(ev(runs[0]), "model_tests.locality.blur.fixed_masks_outside_pred_struct"):
            cd.append(agg(runs, "model_tests.locality.blur.fixed_masks_outside_pred_struct.concept_abs_delta",
                          f"fig2b {m}"))
            tv.append(agg(runs, "summary.fixedmask_task_tv_outside_lesion_blur", f"fig2b {m}"))
        else:
            cd.append(agg(runs, "model_tests.locality.blur.mean_outside_all", f"fig2b {m}"))
            tv.append(agg(runs, "model_tests.locality.blur.task_tv_outside_lesion", f"fig2b {m}"))
    xx = np.arange(len(mk))
    w = 0.38
    for j, m in enumerate(mk):
        col = METHODS.get(m, (m, MUTED, False))[1]
        for off, val, hatch in [(-w / 2 - 0.02, cd[j], None), (w / 2 + 0.02, tv[j], "////")]:
            ax.bar(j + off, val[0], width=w, color=col, alpha=0.5 if hatch else 1.0, hatch=hatch, edgecolor=SURFACE,
                   yerr=val[1] or None, error_kw=dict(lw=0.6, capsize=1.5))
            if np.isfinite(val[0]) and val[0] < 5e-4:
                ax.text(j + off, 0.003, "0", ha="center", va="bottom", fontsize=FS, color=INK, fontweight="bold")
    short = {"cbm_joint": "CBM-\njoint", "cem": "CEM", "sscbm": "SSCBM", "token_pooling": "EB-CBM\n(token)",
             flag_key: "EB-CBM\n(ours)*", "ebcbm": "+CF*",
             "ebcbm_discover": "+disc.*", "ebcbm_proto": "proto*", "ebcbm_proto_patch": "patch\nproto*",
             "ebcbm_no_cf": "vector*"}
    short[flag_key] = "EB-CBM\n(ours)*"
    ax.set_xticks(xx); ax.set_xticklabels([short.get(m, m) for m in mk], fontsize=FS)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("change under blur")
    ax.legend(handles=[Patch_(INK2, "concepts |dp|, outside structures"),
                       Patch_(INK2, "diagnosis TV, outside lesion", hatch=True)],
              loc="upper right", fontsize=FS, borderaxespad=0.1, labelspacing=0.2, handlelength=1.4)
    ax.set_ylim(0, max([v[0] for v in cd + tv if np.isfinite(v[0])] + [0.05]) * (1.45 if big else 1.6))
    ax.set_title("locality (masks frozen)", loc="left", pad=2)
    # (c) CLS ----------------------------------------------------------------------------------------------
    ax = axmm(131.0, yb, 45, ah)
    label(120.5, FH - 0.5, "c")
    n = len(mk)
    w = 0.82 / max(n, 1)
    for j, m in enumerate(mk):
        vals = [agg(R[m], f"model_tests.counterfactual.per_concept.{c}.CLS", f"fig2c {m} {c}")[0] for c in CONCEPTS]
        ax.bar(np.arange(len(CONCEPTS)) - 0.41 + w * (j + 0.5), vals, width=w * 0.88, color=METHODS[m][1])
    ax.axhline(0, color=INK2, lw=0.6)
    abbrev = {"pigment_network": "pigm.\nnetw.", "negative_network": "neg.\nnetw.", "streaks": "streaks",
              "milia_like_cyst": "milia\ncysts", "globules": "glob-\nules"}
    ax.set_xticks(np.arange(len(CONCEPTS))); ax.set_xticklabels([abbrev[c] for c in CONCEPTS], fontsize=FS)
    ax.set_ylabel("CLS"); ax.grid(axis="x", visible=False)
    ax.set_title("counterfactual locality", loc="left", pad=2)
    # shared method legend (b, c)
    from matplotlib.patches import Patch
    disp = {"token_pooling": "EB-CBM (token)"}
    handles = [Patch(color=METHODS[m][1], label=disp.get(m, METHODS[m][0]) + (" *" if METHODS[m][2] else ""))
               for m in mk]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(121.5 / FW, 0.0), ncol=3 if big else len(mk),
               fontsize=FS, handlelength=1.0, columnspacing=0.9, borderaxespad=0.3 if big else 0.15)
    fig.text(11.5 / FW, (1.5 if big else 0.8) / FH, "* strictly local", fontsize=FS, color=INK2)
    save(fig, "fig2_completeness_locality_cls", "Fig. 2: completeness ladder, locality, CLS")
    plt.close(fig)


def Patch_(col, label, hatch=False):
    from matplotlib.patches import Patch
    return Patch(facecolor=col, alpha=0.45 if hatch else 1.0, hatch="////" if hatch else None, edgecolor=SURFACE,
                 label=label)


# ============================================================================ Fig. 3
def _mean_curve(runs, path_x, path_y, what):
    xs, ys = [], []
    for r in runs:
        x, y = dig(ev(r), path_x), dig(ev(r), path_y)
        if x is None or y is None:
            continue
        xs.append(np.array(x, float)); ys.append(np.array(y, float))
    if not ys:
        MISSING.append(f"{what}: {path_y} not found in {runs}")
        return None, None, None
    n = min(len(v) for v in ys)
    Y = np.stack([v[:n] for v in ys])
    return xs[0][:n], Y.mean(0), (Y.std(0, ddof=1) if len(Y) > 1 else None)


def fig3(R: dict[str, list[str]], flag_key: str):
    plt = style()
    FW, FH = PRESET["fig3"]
    FS = PRESET["fs"]
    big = PRESET_NAME == "large"
    fig = plt.figure(figsize=(FW * MM, FH * MM))
    if big:
        axa = fig.add_axes([12 / FW, 72 / FH, 71 / FW, 38 / FH]); axb = fig.add_axes([12 / FW, 12 / FH, 71 / FW, 44 / FH])
    else:
        ah = (FH - 24.0) / 2                          # 23 mm at 70 mm height, 18 mm at 60 mm
        ah_a, ah_b = ah - 2.5, ah + 2.5               # (b) needs room for the line-style legend between the curves
        axa = fig.add_axes([11.5 / FW, (FH - 4.5 - ah_a) / FH, 72.5 / FW, ah_a / FH])
        axb = fig.add_axes([11.5 / FW, 7.5 / FH, 72.5 / FW, ah_b / FH])

    def label(ax, t):
        ax.text(-0.15, 1.14, t, transform=ax.transAxes, fontsize=FS + 1.5, fontweight="bold", va="top", color=INK)
    ax = axa
    label(ax, "a")
    spec = [("blackbox", "-", None), ("cem", "--", None), ("sscbm", ":", None), ("token_pooling", "-.", None),
            (flag_key, "-", "o")]
    ymin = []
    for m, ls, mkr in spec:
        if not R.get(m):
            MISSING.append(f"fig3a: no runs for {m}")
            continue
        x, y, sd = _mean_curve(R[m], "model_tests.deferral.maxprob.curve_cov", "model_tests.deferral.maxprob.curve_acc",
                               f"fig3a {m}")
        if x is None:
            continue
        auc = agg(R[m], "model_tests.deferral.maxprob.auc")[0]
        col = METHODS[m][1]
        name = {"blackbox": "black box", "cem": "CEM", "sscbm": "SSCBM", "token_pooling": "token pool."}.get(m, "ours")
        if big:
            name = {"black box": "Black box", "token pool.": "EB-CBM (token)", "ours": "EB-CBM (ours)"}.get(name, name)
        ax.plot(x, y, ls=ls, color=col, lw=1.3, label=f"{name} {auc:.3f}"[:40],
                marker=mkr, markevery=8, ms=3.0, mec=SURFACE, mew=0.7)
        ymin.append(float(np.min(y[x >= 0.1])))
    ax.set_xlabel("coverage", labelpad=1); ax.set_ylabel("accuracy")
    ax.set_xlim(0.1, 1.0)
    if big:
        lo = ax.get_ylim()[0]
        ax.set_ylim(lo, 1.06)
        ax.set_yticks([t for t in ax.get_yticks() if lo - 1e-9 <= t <= 1.0001])
        ax.legend(fontsize=FS, loc="upper right", ncol=2, columnspacing=0.7, handlelength=1.8, borderaxespad=0.05,
                  labelspacing=0.15)
    else:                       # legend in the empty band below the curves
        lo = np.floor((min(ymin) - 0.23) * 20) / 20 if ymin else 0.45
        ax.set_ylim(lo, 1.0)
        ax.set_yticks([t for t in np.arange(0.0, 1.0001, 0.1) if t >= lo + 0.049])
        ax.legend(fontsize=FS, loc="lower left", ncol=3, columnspacing=0.8, handlelength=1.8, borderaxespad=0.1,
                  labelspacing=0.15, handletextpad=0.4)
    ax.set_title("deferral by max-prob (legend: AUC)", loc="left", pad=2)
    ax = axb
    label(ax, "b")
    ends = []
    for m in [flag_key, "cbm_joint"]:
        if not R.get(m):
            MISSING.append(f"fig3b: no runs for {m}")
            continue
        col = METHODS[m][1]
        for path, ls, lab, mkr in [("model_tests.tti.random", "-", "concepts, perfect expert", "o"),
                                   ("model_tests.tti.random_noisy80", "--", "concepts, expert p=0.8", "s"),
                                   ("model_tests.spatial.random", ":", "masks, perfect expert", "^")]:
            if path.startswith("model_tests.spatial") and dig(ev(R[m][0]), "model_tests.spatial") is None:
                continue                                  # model without masks: no spatial intervention
            x, y, sd = _mean_curve(R[m], path + ".k", path + ".bacc", f"fig3b {m}")
            if x is None:
                continue
            ax.plot(x, y, ls=ls, color=col, marker=mkr, ms=2.8, mec=SURFACE, mew=0.5)
            if path.endswith("tti.random"):
                ends.append((m, x[-1], y[-1], y[0], y[1] if len(y) > 1 else y[0]))
    for m, xe, ye, y0, y1 in ends:                        # direct method labels (colour = method)
        if m == flag_key:
            ax.text(2.5, max(y0, ye) + 0.015, "EB-CBM (ours)", fontsize=FS, color=INK, va="bottom", ha="center")
        else:
            ax.text(0.1, y1 - 0.06, "CBM-joint", fontsize=FS, color=INK, va="top")
    if ends and not big:
        top = max(max(e[3], e[2]) for e in ends)
        ax.set_ylim(ax.get_ylim()[0], top + 0.07)
    style_h = [plt.Line2D([], [], color=INK2, ls=ls, marker=mk, ms=2.8, lw=1.2, label=lab)
               for ls, mk, lab in [("-", "o", "concepts, perfect"), ("--", "s", "concepts, p=0.8"),
                                   (":", "^", "masks, perfect")]]
    if big:
        ax.legend(handles=style_h, fontsize=FS, loc="center right", bbox_to_anchor=(1.0, 0.6), handlelength=2.2,
                  borderaxespad=0.05, labelspacing=0.15)
    else:                                                 # just below the (flat) EB-CBM lines
        lo_, hi_ = ax.get_ylim()
        flat = min(min(e[2], e[3]) for e in ends if e[0] == flag_key) if any(e[0] == flag_key for e in ends) else hi_
        ax.legend(handles=style_h, fontsize=FS, loc="upper right",
                  bbox_to_anchor=(1.0, (flat - 0.025 - lo_) / (hi_ - lo_)), handlelength=2.2, borderaxespad=0.0,
                  labelspacing=0.1, handletextpad=0.5)
    MISSING.append("fig3b: spatial intervention with a noisy expert is not implemented in the evaluator "
                   "(only concept-level noisy expert)")
    ax.set_xlabel("structures corrected (random order)", labelpad=1); ax.set_ylabel("BACC")
    ax.set_title("test-time interventions", loc="left", pad=2)
    save(fig, "fig3_deferral_interventions", "Fig. 3: deferral and interventions")
    plt.close(fig)


# ============================================================================ Table 1
TABLE_COLS = [("BACC", "summary.test_bacc", "max"), ("AUROC", "summary.test_auroc_macro", "max"),
              ("cAUROC", "summary.test_concept_auroc", "max"), ("Dice", "summary.test_dice", "max"),
              ("OIS", "summary.ois", "min"), ("Flip", "summary.artefact_flip_rate", "min"),
              ("PH2", "summary.ph2_auroc", "max"), ("Loc.TV", None, "min")]
TABLE_BASE = ["blackbox", "cbm_joint", "cem", "pcbmh_dino", "lfcbm_clip", "labo", "explicd", "sscbm", "multitask",
              "token_pooling"]


def table1(R: dict[str, list[str]]):
    """Table 1 (booktabs table*, footnotesize, fits the text width). The --flagship_variant row is 'EB-CBM (ours)';
    the other v3-family variants follow as ablation rows only with --show_variants."""
    import csv
    order = TABLE_BASE + [FLAG] + variants(R)
    rows = []
    for key in order:
        name, _, strict = METHODS[key]
        if key in V3_FAMILY and key != FLAG:
            name = "\\quad " + variant_label(key)                 # ablation row under EB-CBM (ours)
        elif key == "token_pooling":
            name = "Token pooling (approx.\\ local)"
        elif key in CITE:
            name = f"{name}~\\cite{{{CITE[key]}}}"
        runs = R.get(key, [])
        ent = {"key": key, "name": name, "strict": strict, "runs": runs, "vals": {}}
        for col, path, _ in TABLE_COLS:
            if not runs:
                ent["vals"][col] = (float("nan"), float("nan"), 0)
                continue
            if col == "Loc.TV":
                v = agg(runs, "summary.fixedmask_task_tv_outside_lesion_blur")
                if v[2] == 0:
                    v = agg(runs, "summary.task_tv_outside_lesion_blur")
                    ent["loc_free"] = v[2] > 0
                ent["vals"][col] = v
            else:
                ent["vals"][col] = agg(runs, path)
        if not runs:
            MISSING.append(f"table1: no runs for {METHODS[key][0]} ({key})")
        else:
            miss = [c for c in ent["vals"] if ent["vals"][c][2] == 0]
            if miss:
                MISSING.append(f"table1: {METHODS[key][0]}: no {', '.join(miss)} in {runs}")
        rows.append(ent)
    best = {}
    main_rows = [r for r in rows if not (r["key"] in V3_FAMILY and r["key"] != FLAG)]   # ablations not ranked
    for col, _, direction in TABLE_COLS:
        vals = [(r["vals"][col][0], r["key"]) for r in main_rows if np.isfinite(r["vals"][col][0])]
        if vals:     # ties at the printed precision are all bold
            bv = round((max if direction == "max" else min)(vals)[0], 3)
            best[col] = {k for v, k in vals if round(v, 3) == bv}

    def num(v):
        t = f"{v:.3f}"
        return t[1:] if t.startswith("0.") else (("-" + t[2:]) if t.startswith("-0.") else t)

    def fmt(r, col):
        m, s, n = r["vals"][col]
        if not np.isfinite(m):
            return "--"
        txt = num(m) if n <= 1 else f"{num(m)}{{\\scriptsize$\\pm${num(s)}}}"
        if col == "Loc.TV" and r.get("loc_free"):
            txt += "\\smash{$^\\dagger$}"        # smashed: the superscript must not increase the row height
        return f"\\textbf{{{txt}}}" if r["key"] in best.get(col, set()) else txt
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    ns = sorted({len(r["runs"]) for r in rows if r["runs"]})
    if not ns:
        seed_note = "no runs"
    elif ns == [1]:
        seed_note = "one seed"
    elif len(ns) == 1:
        seed_note = f"mean $\\pm$ s.d.\\ over {ns[0]} seeds"
    else:
        seed_note = f"mean $\\pm$ s.d.\\ over {ns[-1]} seeds ({min(ns)} for some rows)"
    caption = (f"ISIC 2018 test set ($n{{=}}966$), {seed_note}. Local: every head input is computed inside a predicted "
               "mask. cAUROC: concept AUROC; Flip: artefact flip rate; PH2: melanoma vs.\\ nevus AUROC on PH2; Loc.TV: "
               "diagnosis change under blur outside the predicted lesion, masks frozen ($^\\dagger$: no masks, expert "
               "lesion). Bold: best (lower is better for OIS, Flip, Loc.TV); --: not applicable.")
    sd0 = [METHODS[r["key"]][0] for r in rows if r["key"] in ("labo", "pcbmh_dino") and len(r["runs"]) > 1]
    if sd0:      # their concept columns (LaBo: all columns) do not vary with the seed
        caption += f" {', '.join(sorted(sd0))}: s.d.\\ 0 from frozen features and deterministic solvers."
    lines = ["% generated by src/cbm/scripts/make_figures.py - do not edit by hand",
             "\\begin{table*}[t]", "\\centering", f"\\{TABLE_FONT}", "\\setlength{\\tabcolsep}{3.5pt}",
             "\\caption{" + caption + "}", "\\label{tab:main}",
             "\\begin{tabular}{@{}l c " + "c" * len(TABLE_COLS) + "@{}}", "\\toprule",
             "Method & Local & " + " & ".join(c for c, _, _ in TABLE_COLS) + " \\\\", "\\midrule"]
    for r in rows:
        nm = r["name"].replace("+", "$+$") if "\\cite" not in r["name"] else r["name"]
        if r["key"] == FLAG:
            nm = "\\method{} (ours)"
        lines.append(f"{nm} & {'\\checkmark' if r['strict'] else ''} & " +
                     " & ".join(fmt(r, c) for c, _, _ in TABLE_COLS) + " \\\\")
        if r["key"] == "multitask":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""]
    out = TAB_DIR / "table1_main.tex"
    out.write_text("\n".join(lines))
    OUTPUTS.append(str(out))
    csv_p = ROOT / "results" / "table1_main.csv"
    with open(csv_p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "strictly_local", "n_seeds", "run_ids"] +
                   [f"{c}_{s}" for c, _, _ in TABLE_COLS for s in ("mean", "sd")])
        for r in rows:
            w.writerow([METHODS[r["key"]][0] if r["key"] not in V3_FAMILY or r["key"] == FLAG
                        else f"EB-CBM, {variant_label(r['key'])}", r["strict"], len(r["runs"]), " ".join(r["runs"])] +
                       [v for c, _, _ in TABLE_COLS for v in r["vals"][c][:2]])
    OUTPUTS.append(str(csv_p))


# ============================================================================ final numbers for the writers
NUM_METRICS = {   # name -> (eval.json path, description)
    "BACC": ("summary.test_bacc", "test balanced accuracy (3 classes)"),
    "AUROC": ("summary.test_auroc_macro", "test macro AUROC (one-vs-rest)"),
    "AUROC_mel": ("summary.test_auroc_mel", "test AUROC melanoma vs rest"),
    "macro_F1": ("summary.test_macro_f1", "test macro F1"),
    "cAUROC": ("summary.test_concept_auroc", "test concept AUROC (mean over the 5 structures)"),
    "Dice": ("summary.test_dice", "test structure-mask Dice (images with the structure present)"),
    "lesion_Dice": ("summary.test_lesion_dice", "test lesion-mask Dice"),
    "pointing": ("summary.test_pointing", "pointing game accuracy"),
    "OIS": ("summary.ois", "oracle impurity score (lower = less leakage between concept representations)"),
    "flip": ("summary.artefact_flip_rate", "artefact flip rate (fraction of predictions changed by artefacts)"),
    "artefact_delta_BACC": ("summary.artefact_delta_bacc", "BACC change under artefacts"),
    "PH2_AUROC": ("summary.ph2_auroc", "PH2 (200 images) AUROC melanoma vs nevus from P(melanoma)"),
    "PH2_BACC": ("summary.ph2_bacc", "PH2 BACC melanoma vs nevus"),
    "deferral_AUC": ("summary.deferral_auc_maxprob", "area under the accuracy-coverage curve, max-prob deferral"),
    "ECE": ("summary.test_ece", "test ECE before temperature scaling"),
    "ECE_ts": ("summary.test_ece_ts", "test ECE after temperature scaling (T fit on val NLL)"),
    "NLL_ts": ("summary.test_nll_ts", "test NLL after temperature scaling"),
    "temperature": ("summary.temperature", "fitted temperature"),
    "locTV_free": ("summary.task_tv_outside_lesion_blur",
                   "diagnosis TV under blur outside the lesion, masks recomputed (expert lesion for mask-free models)"),
    "locTV_fixed": ("summary.fixedmask_task_tv_outside_lesion_blur",
                    "diagnosis TV under blur outside the predicted lesion, masks frozen (mask models only)"),
    "loc_concept_delta_free": ("model_tests.locality.blur.mean_outside_all",
                               "mean |dp| of concept probabilities under blur outside all expert structures"),
    "loc_concept_delta_fixed": ("summary.fixedmask_concept_delta_outside_struct_blur",
                                "mean |dp| of concept probabilities under blur outside predicted structures, masks frozen"),
    "CLS": ("summary.CLS", "counterfactual locality score, mean over LaMa structure removals"),
    "XE": ("summary.XE", "cross-effect of a structure removal on the other concepts, mean"),
    "leakage_soft_minus_hard": ("summary.leak_soft_minus_hard_cv", "leakage probe: label-from-soft minus -hard (5-fold CV)"),
    "havasi_soft_minus_hard": ("summary.havasi_soft_minus_hard", "BACC with soft minus hard concepts through the head"),
}


def _stat(runs: list[str], path: str) -> dict | None:
    per = {r: dig(ev(r), path) for r in runs}
    per = {r: float(v) for r, v in per.items() if isinstance(v, (int, float)) and np.isfinite(v)}
    if not per:
        return None
    v = np.array(list(per.values()))
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": len(v),
            "per_seed": {r: round(x, 6) for r, x in per.items()}}


def _stat_vals(per: dict[str, float]) -> dict | None:
    per = {r: float(v) for r, v in per.items() if v is not None and np.isfinite(v)}
    if not per:
        return None
    v = np.array(list(per.values()))
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": len(v),
            "per_seed": {r: round(x, 6) for r, x in per.items()}}


def _curve_delta(runs: list[str], path: str) -> dict | None:
    """BACC after correcting all K structures minus BACC without intervention (k=0), plus the mean curve."""
    per, curves, ks = {}, [], None
    for r in runs:
        c = dig(ev(r), path)
        if not c or "bacc" not in c:
            continue
        b = np.array(c["bacc"], float)
        per[r] = float(b[-1] - b[0]); curves.append(b); ks = c.get("k", list(range(len(b))))
    if not per:
        return None
    n = min(len(c) for c in curves)
    C = np.stack([c[:n] for c in curves])
    out = {"delta_bacc_all_k": _stat_vals(per), "k": list(ks)[:n], "bacc_mean": C.mean(0).round(6).tolist(),
           "bacc_sd": (C.std(0, ddof=1) if len(C) > 1 else np.zeros(n)).round(6).tolist()}
    for extra in ("concept_auroc", "auroc"):
        cc = [np.array(dig(ev(r), path)[extra], float) for r in runs if dig(ev(r), f"{path}.{extra}")]
        if cc:
            m = min(len(c) for c in cc)
            out[f"{extra}_mean"] = np.stack([c[:m] for c in cc]).mean(0).round(6).tolist()
    return out


def final_numbers(R: dict[str, list[str]], flagship_run: str, out_path: Path | None = None):
    """results/final_numbers.json: every number a writer needs (3-seed mean/sd/per-seed, eval.json paths)."""
    import datetime
    out_path = out_path or ROOT / "results" / "final_numbers.json"
    flag_runs = R.get(FLAG, [])
    res = {"generated_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
           "generator": "src/cbm/scripts/make_figures.py (final_numbers)",
           "conventions": {"stat": "mean and sample s.d. (ddof=1) over seeds; per_seed lists the run ids",
                           "test_set": "ISIC 2018 test (n=966) unless stated; PH2: all 200 images",
                           "metrics": {k: {"path": p, "description": d} for k, (p, d) in NUM_METRICS.items()},
                           "official_baselines": "CBM-joint and CEM rows = official-recipe runs bl_cbm_joint_s*, "
                                                 "bl_cem_s*; in-house configs under *_inhouse (internal only)"
                           if USE_OFFICIAL_BL else "CBM-joint and CEM rows = in-house configs"},
           "flagship": {"method_key": FLAG, "name": "EB-CBM (ours)", "runs": flag_runs,
                        "example_run": flagship_run,
                        "example_run_selection": "best validation BACC among the flagship seeds",
                        "val_bacc": {r: dig(ev(r), "val.task.bacc") for r in flag_runs},
                        "ablations": {k: variant_label(k) for k in variants(R)}},
           "methods": {}}
    for key, runs in R.items():
        name = METHODS.get(key, (key, None, None))[0]
        if key in V3_FAMILY and key != FLAG:
            name = f"EB-CBM, {variant_label(key)}"
        ent = {"name": name, "strictly_local": bool(METHODS.get(key, ("", "", False))[2]), "runs": runs,
               "n_seeds": len(runs), "in_table": key in TABLE_BASE + [FLAG] + variants(R), "metrics": {}}
        for mname, (path, _) in NUM_METRICS.items():
            st = _stat(runs, path)
            if st is not None:
                ent["metrics"][mname] = st
        tv = ent["metrics"].get("locTV_fixed") or ent["metrics"].get("locTV_free")
        if tv:
            ent["metrics"]["locTV_table"] = dict(tv, source="fixed" if "locTV_fixed" in ent["metrics"] else "free")
        res["methods"][key] = ent
    # ladder (EB-CBM, 3-seed means) ---------------------------------------------------------------------------------
    lad = {"method": FLAG, "runs": flag_runs, "protocol": "logistic-regression refit on cumulative head-input groups "
           "(train -> test BACC, C on val); head_zeroed = trained head with the other inputs set to 0", "steps": []}
    steps = defaultdict(lambda: defaultdict(dict))
    for r in flag_runs:
        for st in (ev(r).get("ladder") or {}).get("steps", []):
            g = tuple(st["groups"])
            for f in ("refit", "head_zeroed", "head_mean_imputed", "n_inputs"):
                if f in st:
                    steps[g][f][r] = st[f]
    for g in sorted(steps, key=len):
        lad["steps"].append({"groups": list(g), "label": LADDER_LABEL.get(g[-1], g[-1]),
                             "n_inputs": next(iter(steps[g]["n_inputs"].values()), None),
                             **{f: _stat_vals(steps[g][f]) for f in ("refit", "head_zeroed", "head_mean_imputed")}})
    sg = defaultdict(dict)
    for r in flag_runs:
        for k, v in ((ev(r).get("ladder") or {}).get("single_group_refit") or {}).items():
            sg[k][r] = v
    lad["single_group_refit"] = {k: _stat_vals(v) for k, v in sg.items()}
    disc = R.get("ebcbm_discover", [])
    dsteps = defaultdict(dict)
    for r in disc:
        for st in (ev(r).get("ladder") or {}).get("steps", []):
            if "part" in st["groups"]:
                dsteps[tuple(st["groups"])][r] = st["refit"]
    if dsteps:
        g = max(dsteps, key=len)
        lad["discover_variant_parts_step"] = {"method": "ebcbm_discover", "runs": disc, "groups": list(g),
                                              "refit": _stat_vals(dsteps[g])}
    res["ladder"] = lad
    # sanity checks ---------------------------------------------------------------------------------------------------
    san = {}
    for key, runs in R.items():
        d = defaultdict(dict)
        for r in runs:
            s_ = ev(r).get("sanity") or {}
            if not s_.get("available"):
                continue
            for k, v in s_.items():
                if isinstance(v, dict):
                    val = v.get("test_bacc", v.get("mean"))
                    if val is not None:
                        d[k][r] = val
                elif k in ("model_head_test_bacc", "chance_bacc"):
                    d[k][r] = v
        if d:
            san[key] = {k: _stat_vals(v) for k, v in d.items()}
    res["sanity"] = {"protocol": (ev(flag_runs[0]).get("sanity") or {}).get("protocol") if flag_runs else None,
                     "per_method": san}
    # ground-truth concept interventions ---------------------------------------------------------------------------
    gti = {}
    for key in ["cbm_joint", "cem", "sscbm", FLAG, "cbm_joint_inhouse", "cem_inhouse", "token_pooling"] + variants(R):
        runs = R.get(key, [])
        if not runs:
            continue
        gti[key] = {"name": res["methods"][key]["name"], "runs": runs,
                    "perfect_expert_random_order": _curve_delta(runs, "model_tests.tti.random"),
                    "noisy_expert_p0.8_random_order": _curve_delta(runs, "model_tests.tti.random_noisy80"),
                    "perfect_expert_uncertainty_order": _curve_delta(runs, "model_tests.tti.uncertainty"),
                    "havasi_bacc_soft": _stat(runs, "model_tests.havasi.bacc_soft"),
                    "havasi_bacc_gt": _stat(runs, "model_tests.havasi.bacc_gt"),
                    "havasi_intervention_gain": _stat(runs, "model_tests.havasi.intervention_gain")}
    res["gt_interventions"] = {"definition": "delta_bacc_all_k = BACC with all 5 concepts set to the expert labels "
                               "minus BACC without intervention (k=0); curves over k corrected structures",
                               "per_method": gti}
    # spatial (mask) interventions ----------------------------------------------------------------------------------
    sp = {}
    for key in [FLAG] + variants(R) + ["token_pooling"]:
        runs = [r for r in R.get(key, []) if dig(ev(r), "model_tests.spatial")]
        if not runs:
            continue
        allp = {r: dig(ev(r), "model_tests.spatial.all_planes_expert.bacc") - dig(ev(r), "model_tests.spatial.random.bacc")[0]
                for r in runs if dig(ev(r), "model_tests.spatial.all_planes_expert.bacc") is not None}
        sp[key] = {"name": res["methods"][key]["name"], "runs": runs,
                   "random_order": _curve_delta(runs, "model_tests.spatial.random"),
                   "uncertainty_order": _curve_delta(runs, "model_tests.spatial.uncertainty"),
                   "all_planes_expert_delta_bacc": _stat_vals(allp),
                   "all_planes_expert_bacc": _stat(runs, "model_tests.spatial.all_planes_expert.bacc"),
                   "all_planes_expert_concept_auroc": _stat(runs, "model_tests.spatial.all_planes_expert.concept_auroc")}
    res["spatial_interventions"] = {"definition": "expert structure masks replace predicted ones (5 structure "
                                    "planes; all_planes_expert also replaces the lesion plane); delta = BACC after "
                                    "all corrections minus BACC without intervention", "per_method": sp}
    # oracle ceilings -------------------------------------------------------------------------------------------------
    orc = {"definition": "linear head (logistic regression, same protocol as sanity) on expert concept labels "
           "(oracle_concepts) or labels + sqrt(area/lesion) + lesion area (oracle_concepts_areas); "
           "independent of the model up to the shared protocol", "per_method": {}}
    for key in [FLAG, "cbm_joint", "cem", "sscbm"]:
        runs = R.get(key, [])
        if runs:
            orc["per_method"][key] = {"oracle_concepts": _stat(runs, "sanity.oracle_concepts.test_bacc"),
                                      "oracle_concepts_areas": _stat(runs, "sanity.oracle_concepts_areas.test_bacc"),
                                      "chance": _stat(runs, "sanity.chance_bacc")}
    orc["blackbox_bacc"] = _stat(R.get("blackbox", []), "summary.test_bacc")
    orc["flagship_bacc"] = _stat(flag_runs, "summary.test_bacc")
    res["oracle_ceilings"] = orc
    # table 1 as printed + paper status -------------------------------------------------------------------------------
    t1 = ROOT / "results" / "table1_main.csv"
    if t1.exists():
        import csv
        res["table1_csv"] = list(csv.DictReader(open(t1)))
    try:
        import PIL
        import torch
        res["environment"] = f"PIL {PIL.__version__} numpy {np.__version__} torch {torch.__version__}"
    except Exception:
        pass
    res["missing"] = list(dict.fromkeys(m for m in MISSING if not m.startswith("fig")))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(res, indent=1, default=float))
    os.replace(tmp, out_path)
    OUTPUTS.append(str(out_path))


# ============================================================================ main
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--flagship", default="ebcbm_s0")
    ap.add_argument("--prefixes", default="", help="optional stage prefixes of run ids, e.g. A,B,bl (default: none)")
    ap.add_argument("--example", default=None, help="test image id for Fig. 1 (default: chosen automatically)")
    ap.add_argument("--only", nargs="*", default=None, help="subset of fig1 fig1b fig2 fig3 table1 numbers")
    ap.add_argument("--proto_run", default=None, help="crop_backbone_proto run for Fig. 1b (prototype panel)")
    ap.add_argument("--preset", default="isbi", choices=list(PRESETS),
                    help="figure sizes: isbi = main.tex slots (4-page body), isbi_spec = 65/55/70 mm, large = roomy")
    ap.add_argument("--flagship_variant", default=None, choices=list(V3_FAMILY),
                    help="v3-family method shown as 'EB-CBM (ours)' (default: the method of --flagship)")
    ap.add_argument("--show_variants", action="store_true",
                    help="add ablation rows under EB-CBM (ours) (ABLATIONS[flagship], else the other v3 variants)")
    ap.add_argument("--fig_variants", action="store_true", help="also add the ablation variants as Fig. 2b/c bars")
    ap.add_argument("--table_font", default="scriptsize", choices=["scriptsize", "footnotesize", "small"])
    ap.add_argument("--inhouse_cbm", action="store_true",
                    help="CBM-joint / CEM rows from the in-house configs instead of the official-recipe bl_ runs")
    a = ap.parse_args(argv)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")          # figures run on the CPU
    global PRESET_NAME, PRESET, FIG_VARIANTS, USE_OFFICIAL_BL
    PRESET_NAME, PRESET = a.preset, PRESETS[a.preset]
    global TABLE_FONT
    FIG_VARIANTS, USE_OFFICIAL_BL, TABLE_FONT = a.fig_variants, not a.inhouse_cbm, a.table_font
    prefixes = [p.strip() for p in a.prefixes.split(",") if p.strip()]
    R = discover_runs(prefixes)
    flag_key = a.flagship_variant or method_key(a.flagship)[1]
    set_flagship(flag_key, a.show_variants)
    if method_key(a.flagship)[1] != flag_key:          # Fig. 1 example from the chosen variant
        if R.get(flag_key):
            a.flagship = R[flag_key][0]
        else:
            MISSING.append(f"no runs for flagship variant {flag_key}")
    print("[make_figures] methods/runs:", json.dumps(R, indent=0))
    todo = a.only or (["fig1", "fig2", "fig3", "table1", "numbers"] + (["fig1b"] if a.proto_run else []))
    for name in todo:
        try:
            if name == "fig1":
                iid = fig1(a.flagship, a.example)
                print(f"[make_figures] fig1 example: {iid}")
            elif name == "fig1b":
                if not a.proto_run:
                    MISSING.append("fig1b: pass --proto_run <crop_backbone_proto run>")
                else:
                    print(f"[make_figures] fig1b example: {fig1_proto(a.proto_run, a.example)}")
            elif name == "fig2":
                fig2(R, flag_key)
            elif name == "fig3":
                fig3(R, flag_key)
            elif name == "table1":
                table1(R)
            elif name == "numbers":
                final_numbers(R, a.flagship)
        except Exception:
            import traceback
            MISSING.append(f"{name} FAILED: {traceback.format_exc()[-800:]}")
    print("[make_figures] outputs:")
    for o in OUTPUTS:
        print("  ", o)
    if MISSING:
        print("[make_figures] MISSING:")
        for m in dict.fromkeys(MISSING):
            print("  -", m)


if __name__ == "__main__":
    main()
