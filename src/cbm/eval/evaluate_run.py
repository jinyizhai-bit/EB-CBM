"""Evaluator: every DESIGN Section 7 metric from the prediction dumps + the checkpoint.

    python -m cbm.eval.evaluate_run --run_id ebcbm_s0 [--gpu 0] [--fast] [--tests ...] [--merge]

Dump-based (val + test): task metrics (ACC, BACC, macro-F1, AUROC macro / melanoma, ECE, NLL), per-concept
AUROC/F1, grounding (Dice / IoU / pointing vs expert masks at 64x64), leakage probe (label from soft vs hard
concept scores, CV), OIS (AAAI 2023), max-prob / entropy deferral.
Model-based (test; reloads the model and calls the contract API, using cached backbone features when the
model exposes ``forward_heads``):
  havasi       accuracy with GT / hard / soft concepts fed to the head (intervention gap, soft-hard gap)
  tti          test-time intervention curves: random (5 seeds) / uncertainty / verification policies,
               perfect expert and noisy expert (correct with p=0.8, 3 seeds); area under each curve
  spatial      replace k predicted concept masks by expert masks (random x3 / uncertainty order) + all planes
  deferral     accuracy-coverage AUC for max-prob, entropy and influence-weighted verification scores
  locality     Raman-style: blur / noise outside all structure masks vs inside each structure; outside lesion
  counterfactual  CLS_c, XE_c, diagnosis shift under structure removal (needs cf_512 test edits)
  artefacts    ruler / ink / vignette / hair outside the lesion: flip rate, delta BACC, concept drift
  ph2          external PH2 melanoma-vs-nevus AUROC/BACC + concept transfer (from preds_ph2.npz); ``--tests ph2``
               (re)creates preds_ph2.npz through the model API (all 200 PH2 images; any model that follows the model API)
  calibration  (eval.json["calibration"]) post-hoc temperature scaling fitted on the val dump (scalar T, val NLL),
               test ECE / NLL before and after (summary: test_ece_ts, test_nll_ts, temperature); dump-only
  sanity       (eval.json["sanity"]) DESIGN Section 11: retrain only a linear (logistic) head on train-split inputs
               - the model's own head inputs / own concept probabilities, Gaussian random inputs of the same
               dimensionality, label-shuffled inputs, oracle GT concepts [+ GT areas + GT lesion area]; test BACC
  ladder       (eval.json["ladder"]) test BACC with only cumulative subsets of the head-input groups
               (concept, +area, +report, +appearance, +lesion, ...): refit logistic regression on the subset, and
               the model's own head with the other inputs zeroed / replaced by their train mean
``--tests sanity ladder --merge`` recomputes only those sections and merges them into an existing eval.json.
Writes experiments/runs/<run_id>/eval.json and figures/<run_id>/*.png (registered with T.add_figure).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from cbm import track as T  # noqa: E402
from cbm.datamod import (AugParams, DermDataset, collate, dataset_kwargs, denormalize, make_loader,  # noqa: E402
                         normalize)
from cbm.eval import metrics as M  # noqa: E402
from cbm.eval.artefacts import ARTEFACTS, apply_artefact  # noqa: E402

FIG_ROOT = T.ROOT / "figures"
from cbm.paths import CF_INDEX as _CF_INDEX  # noqa: E402

CF_INDEX = str(_CF_INDEX)
EXPECTED_SHIFTS = [  # (removed concept, class, expected sign of delta P(class)) - clinical prior (two-step rule)
    ("milia_like_cyst", "seborrheic_keratosis", -1), ("streaks", "melanoma", -1),
    ("negative_network", "melanoma", -1), ("pigment_network", "seborrheic_keratosis", +1)]


def _f(x):
    """json-safe float."""
    if isinstance(x, (np.floating, float)):
        return None if not np.isfinite(x) else float(x)
    return x


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return _f(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _dc(cfg: dict) -> dict:
    """data config with defaults (baseline configs may not carry every key)."""
    return {"dataset": "isic2018_task12", "image_size": 448, "mask_size": 128, "num_workers": 8,
            **(cfg.get("data") or {})}


# ============================================================================ cache + runner
class Cache:
    """Test images (normalised, bf16 on GPU), GT masks at 128 (uint8), labels, concepts, optional features."""

    def __init__(self, model, ds: DermDataset, device="cuda", bs=32, num_workers=8):
        self.device, self.bs = device, bs
        self.use_heads = hasattr(model, "forward_heads") and hasattr(model, "backbone")
        dl = make_loader(ds, bs, False, num_workers)
        xs, fs, ms, ys, cs, vs, ids = [], [], [], [], [], [], []
        model.eval()
        with torch.no_grad():
            for b in dl:
                x = b["image"].to(device, non_blocking=True)
                xs.append(x.to(torch.bfloat16))
                if self.use_heads:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        ff = model.features(x) if hasattr(model, "features") else model.backbone(x)
                        fs.append(ff.to(torch.bfloat16))
                ms.append((b["masks"] * 255 + 0.5).byte())
                ys.append(b["label"]); cs.append(b["concepts"]); vs.append(b["mask_valid"]); ids += list(b["image_id"])
        self.x, self.f = xs, fs
        self.masks = torch.cat(ms)                       # (N,K+1,128,128) uint8 CPU
        self.y = torch.cat(ys).numpy()
        self.c = torch.cat(cs).numpy()
        self.valid = torch.cat(vs).numpy()
        self.ids = ids
        self.N = len(ids)
        self.K = self.c.shape[1]
        self.slices = []
        s = 0
        for x in xs:
            self.slices.append((s, s + len(x)))
            s += len(x)

    def gt_masks(self, s, e, size=None) -> torch.Tensor:
        m = self.masks[s:e].to(self.device).float() / 255.0
        if size and size != m.shape[-1]:
            m = F.interpolate(m, size=(size, size), mode="bilinear", align_corners=False)
        return m

    @torch.no_grad()
    def run(self, model, co: np.ndarray | None = None, mo_fn=None, keep_masks: bool = False) -> dict:
        res = {"y_prob": [], "c_prob": [], "verify": [], "masks": [], "logits": []}
        for bi, (s, e) in enumerate(self.slices):
            x = self.x[bi].float()
            cob = torch.from_numpy(co[s:e]).to(self.device).float() if co is not None else None
            mob = mo_fn(s, e) if mo_fn is not None else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model.forward_heads(self.f[bi], x, cob, mob) if self.use_heads else \
                    model(x, concept_override=cob, mask_override=mob)
            _collect(res, o, keep_masks)
        return _stack(res)


def _collect(res, o, keep_masks=False):
    lg = o["logits"].float()
    res["logits"].append(lg.cpu().numpy())
    res["y_prob"].append(F.softmax(lg, -1).cpu().numpy())
    res["c_prob"].append(torch.sigmoid(o["concept_logits"].float()).cpu().numpy())
    ex = o.get("extra", {}) or {}
    if "verify" in ex:
        res["verify"].append(ex["verify"].float().cpu().numpy())
    if "lesion_emb" in ex:
        res.setdefault("lesion_emb", []).append(ex["lesion_emb"].float().cpu().numpy())
    if "part_app" in ex:
        res.setdefault("part_app", []).append(ex["part_app"].float().cpu().numpy())
    if keep_masks and "head_inputs" in ex:
        res.setdefault("head_inputs", []).append(ex["head_inputs"].float().cpu().numpy())
    if keep_masks and o.get("masks") is not None:
        res["masks"].append(F.adaptive_avg_pool2d(o["masks"].float(), (64, 64)).cpu().numpy())


def _stack(res):
    return {k: np.concatenate(v) for k, v in res.items() if v}


@torch.no_grad()
def forward_images(model, x: torch.Tensor, bs: int = 32, keep_masks=False) -> dict:
    """Full forward on a tensor of normalised images (on GPU or CPU)."""
    res = {"y_prob": [], "c_prob": [], "verify": [], "masks": [], "logits": []}
    for s in range(0, len(x), bs):
        xb = x[s:s + bs].to("cuda").float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(xb)
        _collect(res, o, keep_masks)
    return _stack(res)


def _bacc(y, prob):
    return M.balanced_accuracy(y, prob.argmax(1), prob.shape[1])


def _summ(y, r, c_true=None):
    d = {"acc": float((r["y_prob"].argmax(1) == y).mean()), "bacc": _bacc(y, r["y_prob"]),
         "auroc": M.macro_auroc(y, r["y_prob"])}
    if c_true is not None and not np.isnan(r["c_prob"]).all():
        d["concept_auroc"] = M.mean_concept_auroc(c_true, r["c_prob"])
    return d


# ============================================================================ model-based tests
def havasi(cache, model, base):
    co_gt = cache.c.copy()
    r_gt = cache.run(model, co=co_gt)
    r_hard = cache.run(model, co=(base["c_prob"] >= 0.5).astype(np.float32))
    y = cache.y
    out = {"bacc_soft": _bacc(y, base["y_prob"]), "bacc_hard": _bacc(y, r_hard["y_prob"]),
           "bacc_gt": _bacc(y, r_gt["y_prob"]), "auroc_soft": M.macro_auroc(y, base["y_prob"]),
           "auroc_gt": M.macro_auroc(y, r_gt["y_prob"]), "auroc_hard": M.macro_auroc(y, r_hard["y_prob"])}
    out["soft_minus_hard"] = out["bacc_soft"] - out["bacc_hard"]
    out["intervention_gain"] = out["bacc_gt"] - out["bacc_soft"]
    out["gt_below_soft"] = bool(out["bacc_gt"] < out["bacc_soft"])     # Havasi leakage symptom
    return out


def reliability(p: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Verification reliability of the predicted concept state: v if predicted present else 1 - v."""
    return np.where(p >= 0.5, v, 1 - v)


def _override_from_order(order_rank, k, answers):
    co = np.full(answers.shape, np.nan, np.float32)
    sel = order_rank < k
    co[sel] = answers[sel]
    return co


def tti(cache, model, base, n_random=5, n_noisy=3, p_correct=0.8):
    N, K = cache.N, cache.K
    y = cache.y
    gt = cache.c.astype(np.float32)
    policies = {}
    rs = np.random.RandomState(0)
    rand_orders = [np.argsort(rs.rand(N, K), 1) for _ in range(n_random)]
    unc = np.argsort(np.abs(base["c_prob"] - 0.5), 1, kind="stable")
    orders = {"uncertainty": [unc], "random": rand_orders}
    if "verify" in base:
        orders["verification"] = [np.argsort(reliability(base["c_prob"], base["verify"]), 1, kind="stable")]

    def curve(order_list, answers_list):
        cs = []
        for order, ans in zip(order_list, answers_list):
            rank = np.argsort(order, 1)
            pts = []
            for k in range(K + 1):
                r = cache.run(model, co=_override_from_order(rank, k, ans)) if k > 0 else base
                pts.append([_bacc(y, r["y_prob"]), float((r["y_prob"].argmax(1) == y).mean()),
                            M.macro_auroc(y, r["y_prob"])])
            cs.append(pts)
        a = np.array(cs)                         # (S,K+1,3)
        return {"k": list(range(K + 1)), "bacc": a[:, :, 0].mean(0).tolist(), "acc": a[:, :, 1].mean(0).tolist(),
                "auroc": a[:, :, 2].mean(0).tolist(), "bacc_std": a[:, :, 0].std(0).tolist(),
                "area_bacc": float(a[:, :, 0].mean())}
    for name, ol in orders.items():
        policies[name] = curve(ol, [gt] * len(ol))
    # noisy expert: answer = GT with prob p_correct else flipped
    rsn = np.random.RandomState(1)
    noisy_answers = []
    for _ in range(n_noisy):
        flip = rsn.rand(N, K) > p_correct
        noisy_answers.append(np.where(flip, 1 - gt, gt).astype(np.float32))
    for name in ["random", "uncertainty"] + (["verification"] if "verification" in orders else []):
        ol = orders[name]
        ol = [ol[i % len(ol)] for i in range(n_noisy)]
        policies[f"{name}_noisy{int(p_correct * 100)}"] = curve(ol, noisy_answers)
    return policies


def spatial(cache, model, base, n_random=3):
    N, K = cache.N, cache.K
    y = cache.y
    dev = cache.device

    def mo_factory(rank, k, all_planes=False):
        def fn(s, e):
            gt = cache.gt_masks(s, e)                                     # (n,K+1,128,128)
            sel = torch.zeros(e - s, K + 1, dtype=torch.bool, device=dev)
            if all_planes:
                sel[:] = True
            else:
                sel[:, :K] = torch.from_numpy(rank[s:e] < k).to(dev)
            return torch.where(sel[:, :, None, None], gt, torch.full_like(gt, float("nan")))
        return fn
    rs = np.random.RandomState(2)
    orders = {"random": [np.argsort(rs.rand(N, K), 1) for _ in range(n_random)],
              "uncertainty": [np.argsort(np.abs(base["c_prob"] - 0.5), 1, kind="stable")]}
    out = {}
    for name, ol in orders.items():
        cs = []
        for order in ol:
            rank = np.argsort(order, 1)
            pts = []
            for k in range(K + 1):
                r = cache.run(model, mo_fn=mo_factory(rank, k)) if k > 0 else base
                pts.append([_bacc(y, r["y_prob"]), M.macro_auroc(y, r["y_prob"]),
                            M.mean_concept_auroc(cache.c, r["c_prob"])])
            cs.append(pts)
        a = np.array(cs)
        out[name] = {"k": list(range(K + 1)), "bacc": a[:, :, 0].mean(0).tolist(), "auroc": a[:, :, 1].mean(0).tolist(),
                     "concept_auroc": a[:, :, 2].mean(0).tolist(), "area_bacc": float(a[:, :, 0].mean())}
    r_all = cache.run(model, mo_fn=mo_factory(None, 0, all_planes=True))
    out["all_planes_expert"] = _summ(y, r_all, cache.c)
    return out


def deferral(cache, model, base, dump_scores: bool = False):
    y = cache.y
    P = base["y_prob"]
    correct = P.argmax(1) == y
    ent = -(P * np.log(np.clip(P, 1e-8, 1))).sum(1)
    scores = {"maxprob": P.max(1), "neg_entropy": -ent}
    if "verify" in base and not np.isnan(base["c_prob"]).all():
        p, v = base["c_prob"], base["verify"]
        r = reliability(p, v)
        yhat = P.argmax(1)
        z0 = base["logits"][np.arange(len(y)), yhat]
        infl = np.zeros_like(p)
        for c in range(cache.K):
            co = np.full(p.shape, np.nan, np.float32)
            co[:, c] = 1.0 - (p[:, c] >= 0.5)
            rr = cache.run(model, co=co)
            infl[:, c] = np.abs(z0 - rr["logits"][np.arange(len(y)), yhat])
        w = infl / np.clip(infl.sum(1, keepdims=True), 1e-8, None)
        scores["verification"] = (w * r).sum(1)
        scores["verification_x_maxprob"] = scores["verification"] * P.max(1)
    out = {k: M.accuracy_coverage(correct, s) for k, s in scores.items()}
    return out


def _dilate(m: torch.Tensor, r: int) -> torch.Tensor:
    return F.max_pool2d(m, 2 * r + 1, stride=1, padding=r) if r > 0 else m


def _perturb(x01, kind, gen):
    import torchvision.transforms.functional as TF
    if kind == "blur":
        return TF.gaussian_blur(x01, kernel_size=21, sigma=5.0)
    if kind == "noise":
        return (x01 + 0.1 * torch.randn(x01.shape, device=x01.device, generator=gen)).clamp(0, 1)
    raise ValueError(kind)


@torch.no_grad()
def locality(cache, model, base, size=448):
    """Raman-style locality. For each perturbation (blur, noise): (a) outside all expert structure masks
    (union dilated 7 px), (b) inside each structure's own expert mask (dilated 2 px, images where present),
    (c) outside the lesion.  Reports mean |delta p_c| and total-variation change of the diagnosis."""
    K, dev = cache.K, cache.device
    out = {}
    for kind in ["blur", "noise"]:
        gen = torch.Generator(device=dev).manual_seed(0)
        dp_out, dp_les, dy_out, dy_les, dy_lin = [], [], [], [], []
        le_out, le_in, le_norm = [], [], []
        fx_tv, fx_dc, fx_le, fx_ln, fs_dc, fs_tv, fs_mx, fe_dc, fe_mx, fe_n = [], [], [], [], [], [], [], [], [], []
        fp_mx, fp_rel, fp_les, fp_ln = [], [], [], []
        dp_in = {c: [] for c in range(K)}
        dy_in = {c: [] for c in range(K)}
        for bi, (s, e) in enumerate(cache.slices):
            x01 = denormalize(cache.x[bi].float())
            xp = _perturb(x01, kind, gen)
            gt = (cache.gt_masks(s, e, size) > 0.5).float()                       # (n,K+1,S,S)
            U = _dilate(gt[:, :K].amax(1, keepdim=True), 7)
            L = _dilate(gt[:, K:K + 1], 5)
            xs = [U * x01 + (1 - U) * xp, L * x01 + (1 - L) * xp]
            r_out = forward_images(model, normalize(xs[0]))
            r_les = forward_images(model, normalize(xs[1]))
            pb, yb = base["c_prob"][s:e], base["y_prob"][s:e]
            dp_out.append(np.abs(r_out["c_prob"] - pb)); dy_out.append(0.5 * np.abs(r_out["y_prob"] - yb).sum(1))
            dp_les.append(np.abs(r_les["c_prob"] - pb)); dy_les.append(0.5 * np.abs(r_les["y_prob"] - yb).sum(1))
            r_lin = forward_images(model, normalize(L * xp + (1 - L) * x01))      # perturb inside the lesion only
            dy_lin.append(0.5 * np.abs(r_lin["y_prob"] - yb).sum(1))
            if "masks" in base and len(base["masks"]):
                # masks frozen at the model's own (unperturbed) prediction; perturb outside the *predicted* lesion:
                # isolates feature-level leakage (e.g. ViT global context) from changes of the predicted masks
                mo = torch.from_numpy(base["masks"][s:e]).to(dev).float()            # (n,K+1,64,64)
                mo128 = F.interpolate(mo, size=(128, 128), mode="bilinear", align_corners=False)
                # protected region = nearest-upsampled 128-px binarised mask (exactly the region crops may read),
                # dilated 3 px
                Lp = _dilate(F.interpolate((mo128[:, K:K + 1] > 0.5).float(), size=x01.shape[-2:], mode="nearest"), 3)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o_ref = model(normalize(x01), mask_override=mo128)
                    o_fx = model(normalize(Lp * x01 + (1 - Lp) * xp), mask_override=mo128)
                fx_tv.append(0.5 * (F.softmax(o_fx["logits"].float(), -1) - F.softmax(o_ref["logits"].float(), -1))
                             .abs().sum(1).cpu().numpy())
                fx_dc.append((torch.sigmoid(o_fx["concept_logits"].float()) - torch.sigmoid(o_ref["concept_logits"].float()))
                             .abs().mean(1).cpu().numpy())
                # same, perturbing outside the union of the predicted *structure* masks (RR concept evidence)
                Sp = _dilate(F.interpolate((mo128[:, :K] > 0.5).float().amax(1, keepdim=True), size=x01.shape[-2:],
                                           mode="nearest"), 3)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o_fs = model(normalize(Sp * x01 + (1 - Sp) * xp), mask_override=mo128)
                fs_dc.append((torch.sigmoid(o_fs["concept_logits"].float()) - torch.sigmoid(o_ref["concept_logits"].float()))
                             .abs().mean(1).cpu().numpy())
                fs_mx.append(float((o_fs["concept_logits"].float() - o_ref["concept_logits"].float()).abs().max()))
                fs_tv.append(0.5 * (F.softmax(o_fs["logits"].float(), -1) - F.softmax(o_ref["logits"].float(), -1))
                             .abs().sum(1).cpu().numpy())
                # masks frozen to the EXPERT masks, perturb outside the expert structure union: non-trivial crops for
                # every image with an annotated structure, independent of how good the predicted masks are
                g128 = cache.gt_masks(s, e)                                          # (n,K+1,128,128) soft
                Ep = _dilate(F.interpolate((g128[:, :K] > 0.5).float().amax(1, keepdim=True), size=x01.shape[-2:],
                                           mode="nearest"), 3)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o_er = model(normalize(x01), mask_override=g128)
                    o_ep = model(normalize(Ep * x01 + (1 - Ep) * xp), mask_override=g128)
                dce = (o_ep["concept_logits"].float() - o_er["concept_logits"].float()).abs()
                pres = torch.from_numpy(cache.c[s:e] == 1).to(dev)
                fe_dc.append(dce[pres].cpu().numpy())
                fe_mx.append(float(dce.max()))
                fe_n.append(int(pres.sum()))
                if "part_app" in o_ref.get("extra", {}):
                    # discovered parts: perturb outside the union of the part supports (bilinear-upsampled soft part
                    # masks > support_tau, exactly the pixels part crops may read), masks frozen -> drift must be 0
                    tau = float(getattr(model.concept_head, "part_tau", 0.1))
                    up = F.interpolate(mo128[:, K + 1:], size=x01.shape[-2:], mode="bilinear", align_corners=False)
                    Pp = _dilate((up > tau).float().amax(1, keepdim=True), 3)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        o_fp = model(normalize(Pp * x01 + (1 - Pp) * xp), mask_override=mo128)
                    dpa = (o_fp["extra"]["part_app"].float() - o_ref["extra"]["part_app"].float())
                    fp_mx.append(float(dpa.abs().max()))
                    fp_rel.append(dpa.norm(dim=-1).cpu().numpy().ravel())
                    fp_ln.append(o_ref["extra"]["part_app"].float().norm(dim=-1).cpu().numpy().ravel())
                    fp_les.append((o_fx["extra"]["part_app"].float() - o_ref["extra"]["part_app"].float())
                                  .norm(dim=-1).cpu().numpy().ravel())
                if "lesion_emb" in o_ref.get("extra", {}):
                    fx_le.append((o_fx["extra"]["lesion_emb"] - o_ref["extra"]["lesion_emb"]).float().norm(dim=1).cpu().numpy())
                    fx_ln.append(o_ref["extra"]["lesion_emb"].float().norm(dim=1).cpu().numpy())
            if "lesion_emb" in base and "lesion_emb" in r_les:
                eb = base["lesion_emb"][s:e]
                le_out.append(np.linalg.norm(r_les["lesion_emb"] - eb, axis=1))
                le_in.append(np.linalg.norm(r_lin["lesion_emb"] - eb, axis=1))
                le_norm.append(np.linalg.norm(eb, axis=1))
            for c in range(K):
                pres = (cache.c[s:e, c] == 1)
                if not pres.any():
                    continue
                ii = torch.from_numpy(np.flatnonzero(pres)).to(dev)
                E = _dilate(gt[ii, c:c + 1], 2)
                xi = E * xp[ii] + (1 - E) * x01[ii]
                r_in = forward_images(model, normalize(xi))
                dp_in[c].append(np.abs(r_in["c_prob"][:, c] - pb[pres, c]))
                dy_in[c].append(0.5 * np.abs(r_in["y_prob"] - yb[pres]).sum(1))
        dp_out, dp_les = np.concatenate(dp_out), np.concatenate(dp_les)
        res = {"task_tv_outside_struct": float(np.mean(np.concatenate(dy_out))),
               "task_tv_outside_lesion": float(np.mean(np.concatenate(dy_les))),
               "task_tv_inside_lesion": float(np.mean(np.concatenate(dy_lin))), "per_concept": {}}
        if le_out:
            lo, li, ln = np.concatenate(le_out), np.concatenate(le_in), np.concatenate(le_norm)
            # NB: pooling is restricted to lesion positions, but ViT tokens see global context through attention,
            # so outside-lesion drift is measured, not assumed to be zero.
            res["lesion_emb"] = {"rel_drift_outside_lesion": float(lo.mean() / max(ln.mean(), 1e-8)),
                                 "rel_drift_inside_lesion": float(li.mean() / max(ln.mean(), 1e-8)),
                                 "ratio_out_in": float(lo.mean() / max(li.mean(), 1e-8))}
        if fx_tv:
            res["fixed_masks_outside_pred_lesion"] = {
                "task_tv": float(np.mean(np.concatenate(fx_tv))),
                "concept_abs_delta": float(np.nanmean(np.concatenate(fx_dc)))}
            res["fixed_masks_outside_pred_struct"] = {"concept_abs_delta": float(np.nanmean(np.concatenate(fs_dc))),
                                                      "concept_logit_max_abs_delta": float(np.nanmax(fs_mx)),
                                                      "task_tv": float(np.mean(np.concatenate(fs_tv)))}
            if fp_mx:
                nrm = max(np.concatenate(fp_ln).mean(), 1e-8)
                res["fixed_masks_outside_part_support"] = {
                    "part_desc_max_abs_delta": float(np.max(fp_mx)),
                    "part_desc_rel_drift": float(np.concatenate(fp_rel).mean() / nrm),
                    "part_desc_rel_drift_outside_pred_lesion": float(np.concatenate(fp_les).mean() / nrm)}
            res["fixed_expert_masks_outside_struct"] = {
                "concept_logit_mean_abs_delta_present": float(np.mean(np.concatenate(fe_dc))) if sum(fe_n) else 0.0,
                "concept_logit_max_abs_delta": float(np.nanmax(fe_mx)), "n_present_concepts": int(sum(fe_n))}
            if fx_le:
                res["fixed_masks_outside_pred_lesion"]["lesion_emb_rel_drift"] = float(
                    np.concatenate(fx_le).mean() / max(np.concatenate(fx_ln).mean(), 1e-8))
        names = cache_concepts(cache)
        gaps, ratios = [], []
        for c in range(K):
            pres = cache.c[:, c] == 1
            o_all = float(np.nanmean(dp_out[:, c]))
            o_pres = float(np.nanmean(dp_out[pres, c])) if pres.any() else float("nan")
            i_pres = float(np.nanmean(np.concatenate(dp_in[c]))) if dp_in[c] else float("nan")
            res["per_concept"][names[c]] = {"outside_all": o_all, "outside_present": o_pres, "inside": i_pres,
                                            "outside_lesion": float(np.nanmean(dp_les[:, c])),
                                            "task_tv_inside": float(np.mean(np.concatenate(dy_in[c]))) if dy_in[c]
                                            else float("nan"),
                                            "ratio_out_in": o_pres / i_pres if i_pres and i_pres > 0 else float("nan")}
            gaps.append(i_pres - o_pres)
            ratios.append(res["per_concept"][names[c]]["ratio_out_in"])
        res["mean_gap_in_minus_out"] = float(np.nanmean(gaps))
        res["mean_ratio_out_in"] = float(np.nanmean(ratios))
        res["mean_outside_all"] = float(np.nanmean(dp_out))
        out[kind] = res
    return out


def cache_concepts(cache):
    return getattr(cache, "concept_names", [str(i) for i in range(cache.K)])


class _ImgList(torch.utils.data.Dataset):
    def __init__(self, paths, size):
        self.paths, self.size = paths, size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        img = Image.open(self.paths[i]).convert("RGB")
        return normalize(AugParams(*img.size, train=False).image(img, self.size))


@torch.no_grad()
def counterfactual(cache, model, base, size=448, cf_index=CF_INDEX):
    import pandas as pd
    if not os.path.exists(cf_index):
        return {"available": False, "reason": "cf index missing"}
    idx = pd.read_csv(cf_index)
    idx = idx[(idx.split == "test") & idx.image_id.isin(set(cache.ids))]
    if not len(idx):
        return {"available": False, "reason": "no test edits"}
    idx = idx[idx.path.map(os.path.exists)]
    pos = {iid: i for i, iid in enumerate(cache.ids)}
    dl = torch.utils.data.DataLoader(_ImgList(idx.path.tolist(), size), batch_size=32, num_workers=8)
    res = {"y_prob": [], "c_prob": [], "verify": [], "masks": [], "logits": []}
    for xb in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(xb.to("cuda"))
        _collect(res, o, keep_masks=True)
    r = _stack(res)
    names = cache_concepts(cache)
    K = cache.K
    rows = idx.reset_index(drop=True)
    rows["_i"] = rows.image_id.map(pos)
    out = {"available": True, "n_edits": int(len(rows)), "per_concept": {}}
    cls_all, xe_all = [], []
    class_names = getattr(cache, "class_names", [str(i) for i in range(base["y_prob"].shape[1])])
    shifts = {}
    tv_by_img: dict[int, list] = {}
    for c, cn in enumerate(names):
        mi = rows.index[(rows.kind == "minus") & (rows.concept == cn)].to_numpy()
        ct = rows.index[(rows.kind == "ctrl") & (rows.concept == cn)].to_numpy()
        im_m = rows.loc[mi, "_i"].to_numpy()
        im_c = rows.loc[ct, "_i"].to_numpy()
        common = np.intersect1d(im_m, im_c)
        if len(common) == 0:
            continue
        mi = mi[np.isin(im_m, common)]; ct = ct[np.isin(im_c, common)]
        im_m, im_c = rows.loc[mi, "_i"].to_numpy(), rows.loc[ct, "_i"].to_numpy()
        om, oc = np.argsort(im_m), np.argsort(im_c)
        mi, ct, ii = mi[om], ct[oc], im_m[om]
        s0 = base["c_prob"][ii]
        sm, sc = r["c_prob"][mi], r["c_prob"][ct]
        d_rm, d_ct = s0[:, c] - sm[:, c], s0[:, c] - sc[:, c]
        others = [k for k in range(K) if k != c]
        xe = np.abs(s0[:, others] - sm[:, others]).mean(1)
        xe_c = np.abs(s0[:, others] - sc[:, others]).mean(1)
        dP = r["y_prob"][mi] - base["y_prob"][ii]
        dPc = r["y_prob"][ct] - base["y_prob"][ii]
        ent = {"n": int(len(ii)), "removal_drop": float(np.nanmean(d_rm)), "control_drop": float(np.nanmean(d_ct)),
               "CLS": float(np.nanmean(d_rm) - np.nanmean(d_ct)), "XE": float(np.nanmean(xe)),
               "XE_control": float(np.nanmean(xe_c)),
               "frac_switched_off": float(np.nanmean((s0[:, c] >= 0.5) & (sm[:, c] < 0.5)) /
                                          max(np.nanmean(s0[:, c] >= 0.5), 1e-8)),
               "delta_P": {class_names[j]: float(dP[:, j].mean()) for j in range(dP.shape[1])},
               "delta_P_control": {class_names[j]: float(dPc[:, j].mean()) for j in range(dP.shape[1])}}
        if "masks" in r and len(r["masks"]):
            ent["mask_area_drop"] = float((r["masks"][ct][:, c].mean((1, 2)) - r["masks"][mi][:, c].mean((1, 2))).mean())
        out["per_concept"][cn] = ent
        cls_all.append(ent["CLS"]); xe_all.append(ent["XE"])
        shifts[cn] = ent["delta_P"]
        for j, i_img in enumerate(ii):
            tv_by_img.setdefault(int(i_img), []).append((cn, 0.5 * np.abs(dP[j]).sum(), dP[j]))
    out["mean_CLS"] = float(np.nanmean(cls_all)) if cls_all else float("nan")
    out["mean_XE"] = float(np.nanmean(xe_all)) if xe_all else float("nan")
    # most influential concept per image
    top = [max(v, key=lambda t: t[1]) for v in tv_by_img.values()]
    if top:
        out["most_influential"] = {"mean_tv": float(np.mean([t[1] for t in top])),
                                   "counts": {cn: int(sum(t[0] == cn for t in top)) for cn in names}}
    checks = []
    for cn, cl, sign in EXPECTED_SHIFTS:
        if cn in shifts and cl in shifts[cn]:
            checks.append({"concept": cn, "class": cl, "expected_sign": sign, "delta": shifts[cn][cl],
                           "agrees": bool(np.sign(shifts[cn][cl]) == sign)})
    out["clinical_direction_checks"] = checks
    return out


@torch.no_grad()
def artefacts(cache, model, base, size=448):
    K, dev = cache.K, cache.device
    y = cache.y
    out = {}
    examples = {}
    for a_i, kind in enumerate(ARTEFACTS):
        res = {"y_prob": [], "c_prob": [], "verify": [], "masks": [], "logits": []}
        applied = []
        for bi, (s, e) in enumerate(cache.slices):
            x01 = denormalize(cache.x[bi].float())
            les = cache.gt_masks(s, e, size)[:, K]
            xa = apply_artefact(x01, les, kind, seeds=[1000 * a_i + i for i in range(s, e)])
            applied.append(((xa - x01).abs().flatten(1).amax(1) > 2.5 / 255).cpu().numpy())
            if bi == 0:
                examples[kind] = xa[0].cpu()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(normalize(xa))
            _collect(res, o)
        r = _stack(res)
        ent = {"flip_rate": float((r["y_prob"].argmax(1) != base["y_prob"].argmax(1)).mean()),
               "bacc": _bacc(y, r["y_prob"]), "delta_bacc": _bacc(y, r["y_prob"]) - _bacc(y, base["y_prob"]),
               "delta_p_mel": float((r["y_prob"][:, 1] - base["y_prob"][:, 1]).mean()),
               "abs_delta_p_mel": float(np.abs(r["y_prob"][:, 1] - base["y_prob"][:, 1]).mean()),
               "frac_images_modified": float(np.concatenate(applied).mean())}
        if not np.isnan(base["c_prob"]).all():
            ent["concept_abs_drift"] = float(np.abs(r["c_prob"] - base["c_prob"]).mean())
            ent["concept_auroc"] = M.mean_concept_auroc(cache.c, r["c_prob"])
        out[kind] = ent
    out["mean_flip_rate"] = float(np.mean([out[k]["flip_rate"] for k in ARTEFACTS]))
    out["mean_delta_bacc"] = float(np.mean([out[k]["delta_bacc"] for k in ARTEFACTS]))
    return out, examples


@torch.no_grad()
def ph2_dump(run_dir: Path, cfg: dict, model, device: str = "cuda", bs: int = 32) -> Path:
    """External test through the model API (works for every model that follows the contract): all PH2 images of data/processed/ph2 (the same 200 images for every method; PH2 'excluded' rows, if any,
    are dropped exactly as in the core trainer), eval transform at the model's own input size (ImageNet
    normalisation). Writes preds_ph2.npz in the core format."""
    from cbm.data.datasets import DATASETS
    from cbm.datamod import PH2Dataset
    ds = PH2Dataset(_dc(cfg)["image_size"], _dc(cfg)["mask_size"])
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=4, collate_fn=collate)
    model.eval()
    acc = {k: [] for k in ["image_id", "y_true", "y_prob", "c_prob", "ph2_concepts", "mask_pred", "lesion_true"]}
    all_valid = True
    for b in dl:
        x = b["image"].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
            out = model(x)
        acc["image_id"] += list(b["image_id"])
        acc["y_true"].append(b["label"].numpy())
        acc["y_prob"].append(F.softmax(out["logits"].float(), -1).cpu().numpy())
        acc["c_prob"].append(torch.sigmoid(out["concept_logits"].float()).cpu().numpy())
        acc["ph2_concepts"].append(b["ph2_concepts"].numpy())
        all_valid &= bool(b["lesion_valid"].all())
        acc["lesion_true"].append((F.adaptive_avg_pool2d(b["lesion"][:, None].float(), (64, 64))[:, 0] >= 0.5).numpy())
        if out.get("masks") is not None:
            m64 = F.adaptive_avg_pool2d(out["masks"].float(), (64, 64))
            acc["mask_pred"].append((m64 * 255 + 0.5).clamp(0, 255).byte().cpu().numpy())
    d = {k: (np.array(v) if k == "image_id" else np.concatenate(v)) for k, v in acc.items() if v}
    if not all_valid:
        d.pop("lesion_true", None)
    pt = run_dir / "preds_test.npz"
    names = np.load(pt, allow_pickle=True) if pt.exists() else None
    d["ph2_class_names"] = np.array(ds.info["class_names"])
    d["ph2_concept_names"] = np.array(ds.info["concept_names"])
    d["class_names"] = names["class_names"] if names is not None and "class_names" in names.files else \
        np.array(DATASETS["isic2018_task12"]["class_names"])
    d["concept_names"] = names["concept_names"] if names is not None and "concept_names" in names.files else \
        np.array(DATASETS["isic2018_task12"]["concept_names"])
    np.savez_compressed(run_dir / "preds_ph2.npz", **d)
    return run_dir / "preds_ph2.npz"


def ph2_metrics(run_dir: Path) -> dict:
    p = run_dir / "preds_ph2.npz"
    if not p.exists():
        return {"available": False}
    from sklearn.metrics import roc_auc_score
    d = np.load(p, allow_pickle=True)
    ph2_names = list(d["ph2_class_names"])
    mel_ph2 = ph2_names.index("melanoma")
    cls = list(d["class_names"])
    mi = cls.index("melanoma")
    y = (d["y_true"] == mel_ph2).astype(int)
    pm = d["y_prob"][:, mi]
    pred = (d["y_prob"].argmax(1) == mi).astype(int)
    out = {"available": True, "n": int(len(y)), "auroc_mel_vs_nevus": float(roc_auc_score(y, pm)),
           "bacc_mel_vs_nevus": M.balanced_accuracy(y, pred, 2),
           "sens": float(pred[y == 1].mean()), "spec": float(1 - pred[y == 0].mean())}
    cn = list(d["concept_names"]); pcn = list(d["ph2_concept_names"])
    pc = d["ph2_concepts"]
    tr = {}
    maps = {"streaks": ["streaks"], "globules": ["dots_globules_typical", "dots_globules_atypical"]}
    if not np.isnan(d["c_prob"]).all() and d["c_prob"].shape[1] == len(cn):
        for ours, theirs in maps.items():
            if ours in cn and all(t in pcn for t in theirs):
                t = np.nanmax(np.stack([pc[:, pcn.index(x)] for x in theirs], 1), 1)
                ok = ~np.isnan(t)
                if len(np.unique(t[ok])) == 2:
                    tr[ours] = float(roc_auc_score(t[ok], d["c_prob"][ok, cn.index(ours)]))
    out["concept_transfer_auroc"] = tr
    if "mask_pred" in d.files and "lesion_true" in d.files and d["mask_pred"].shape[1] > len(cn):
        pl = d["mask_pred"][:, len(cn)] >= 128              # lesion plane (part planes, if any, come after it)
        tl = d["lesion_true"].astype(bool)
        inter = (pl & tl).reshape(len(pl), -1).sum(1)
        den = pl.reshape(len(pl), -1).sum(1) + tl.reshape(len(tl), -1).sum(1)
        out["lesion_dice"] = float(np.mean(2 * inter / np.maximum(den, 1)))
    return out


# ============================================================================ dump-based
def gt_masks_64(ds: DermDataset) -> dict:
    out = {}
    for i in range(len(ds)):
        from PIL import Image
        ap = AugParams(1, 1, train=False)
        planes = []
        for p in ds.plane_paths[i]:
            if isinstance(p, str) and os.path.exists(p):
                im = Image.open(p).convert("L")
                ap.box = (0, 0) + im.size
                planes.append(ap.mask(im, 64).numpy() >= 0.5)
            else:
                planes.append(np.zeros((64, 64), bool))
        out[ds.ids[i]] = np.stack(planes)
    return out


def dump_eval(run_dir: Path, split: str, cfg: dict) -> dict:
    p = run_dir / f"preds_{split}.npz"
    if not p.exists():
        return {"available": False}
    d = np.load(p, allow_pickle=True)
    cls, cn = list(d["class_names"]), list(d["concept_names"])
    out = {"n": int(len(d["y_true"])),
           "task": M.task_metrics(d["y_true"], d["y_prob"], cls, cls.index("melanoma") if "melanoma" in cls else 1),
           "concepts": M.concept_metrics(d["c_true"], d["c_prob"], cn)}
    if "mask_pred" in d.files:
        ds = DermDataset(_dc(cfg)["dataset"], split, subset=_dc(cfg).get("subset_eval", 0), **dataset_kwargs(cfg))
        gm = gt_masks_64(ds)
        mt = np.stack([gm[i] for i in d["image_id"]])
        out["grounding"] = M.grounding_metrics(d["mask_pred"], mt, cn + ["lesion"])
        if "area_pred" in d.files:
            ok = ~np.isnan(d["area_true"]) & ~np.isnan(d["area_pred"])
            out["area_mae_lesion_frac"] = float(np.abs(d["area_true"][ok] - d["area_pred"][ok]).mean())
    has_c = not np.isnan(d["c_prob"]).all()
    if split == "test" and has_c:
        out["leakage_probe"] = M.leakage_cv(d["c_true"], d["c_prob"], d["y_true"],
                                            d["c_repr"] if "c_repr" in d.files else None)
        rep = d["c_repr"] if "c_repr" in d.files else d["c_prob"][..., None]
        out["ois"] = M.ois(d["c_true"], rep)
    P = d["y_prob"]
    correct = P.argmax(1) == d["y_true"]
    ent = -(P * np.log(np.clip(P, 1e-8, 1))).sum(1)
    out["deferral"] = {"maxprob": M.accuracy_coverage(correct, P.max(1)),
                       "neg_entropy": M.accuracy_coverage(correct, -ent)}
    if "report" in d.files and "report_names" in d.files:
        out["report"] = {"names": list(d["report_names"])}
    return out


def report_weights(model, cfg) -> dict:
    """I3: linear report-head weights (standardised space) + agreement with the clinical rules."""
    if (cfg.get("model") or {}).get("diag_head") != "report" or getattr(getattr(model, "diag", None), "kind", "") != "linear":
        return {}
    W = model.diag.net.weight.detach().float().cpu().numpy()
    names = model.report_names
    cls = ["nevus", "melanoma", "seborrheic_keratosis"]
    w = {cls[j]: {names[i]: float(W[j, i]) for i in range(len(names))} for j in range(W.shape[0])}
    rules = [("milia_like_cyst:presence", "seborrheic_keratosis", +1), ("negative_network:presence", "melanoma", +1),
             ("streaks:presence", "melanoma", +1), ("pigment_network:presence", "seborrheic_keratosis", -1)]
    checks = [{"feature": f, "class": c, "expected_sign": s, "weight": w[c][f],
               "agrees": bool(np.sign(w[c][f]) == s)} for f, c, s in rules if f in w[c]]
    return {"weights": w, "rule_checks": checks, "n_nonzero": int((np.abs(W) > 1e-6).sum())}


# ============================================================================ calibration
def temperature_scaling(run_dir: Path) -> dict:
    """Post-hoc temperature scaling: one scalar T > 0 minimising the val NLL of softmax(logits / T) (logits from the
    dump, or log-probabilities if the dump has none), applied to the test dump. Accuracy / BACC are unchanged."""
    from scipy.optimize import minimize_scalar
    pv, pt = run_dir / "preds_val.npz", run_dir / "preds_test.npz"
    if not (pv.exists() and pt.exists()):
        return {"available": False, "reason": "missing preds_val/test.npz"}
    dv, dt = np.load(pv, allow_pickle=True), np.load(pt, allow_pickle=True)

    def logits(d):
        return d["logits"].astype(np.float64) if "logits" in d.files else np.log(np.clip(d["y_prob"], 1e-8, 1.0))

    def softmax(z):
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def nll(z, y):
        return float(-np.log(np.clip(softmax(z)[np.arange(len(y)), y], 1e-12, 1.0)).mean())
    zv, yv, zt, yt = logits(dv), dv["y_true"].astype(int), logits(dt), dt["y_true"].astype(int)
    res = minimize_scalar(lambda lt: nll(zv / np.exp(lt), yv), bounds=(-3.0, 3.0), method="bounded")
    T = float(np.exp(res.x))
    pt0, pt1 = softmax(zt), softmax(zt / T)
    return {"available": True, "temperature": T, "val_nll": nll(zv, yv), "val_nll_ts": nll(zv / T, yv),
            "test_ece": M.ece_score(yt, pt0), "test_ece_ts": M.ece_score(yt, pt1),
            "test_nll": nll(zt, yt), "test_nll_ts": nll(zt / T, yt),
            "logits_source": "logits" if "logits" in dt.files else "log y_prob"}


# ============================================================================ discovered parts
def parts_report(cache, model, base) -> dict:
    """Mean area, presence, overlap with expert structures / lesion and descriptor statistics of the discovered
    evidence regions (mask planes K+1..)."""
    if "masks" not in base or base["masks"].shape[1] <= cache.K + 1:
        return {"available": False}
    K = cache.K
    m = base["masks"]                                                         # (N,K+1+P,64,64) soft
    P = m.shape[1] - K - 1
    gt = F.adaptive_avg_pool2d(cache.masks.float() / 255.0, (64, 64)).numpy() >= 0.5   # (N,K+1,64,64)
    les_pred = m[:, K]
    les_gt = gt[:, K]
    struct = gt[:, :K]
    union = struct.any(1)
    min_area = float(getattr(model.concept_head, "min_area", 0.002))
    names = cache_concepts(cache)
    out = {"available": True, "n_parts": int(P), "parts": {}}
    eps = 1e-6
    for j in range(P):
        pm = m[:, K + 1 + j]
        mass = pm.reshape(len(pm), -1).sum(1) + eps
        hard = pm >= 0.5
        ent = {"area_frac_pred_lesion": float(np.mean(mass / (les_pred.reshape(len(pm), -1).sum(1) + eps))),
               "area_frac_gt_lesion": float(np.mean(mass / (les_gt.reshape(len(pm), -1).sum(1) + eps))),
               "presence_rate": float(np.mean(hard.reshape(len(pm), -1).mean(1) >= min_area)),
               "mass_inside_gt_lesion": float(np.mean((pm * les_gt).reshape(len(pm), -1).sum(1) / mass)),
               "mass_inside_expert_structures": float(np.mean((pm * union).reshape(len(pm), -1).sum(1) / mass)),
               "soft_dice_with_structures": {}}
        for c in range(K):
            has = struct[:, c].reshape(len(pm), -1).any(1)
            if has.any():
                inter = (pm[has] * struct[has, c]).reshape(int(has.sum()), -1).sum(1)
                den = pm[has].reshape(int(has.sum()), -1).sum(1) + struct[has, c].reshape(int(has.sum()), -1).sum(1)
                ent["soft_dice_with_structures"][names[c]] = float(np.mean(2 * inter / (den + eps)))
        if "part_app" in base:
            pa = base["part_app"][:, j]
            ent["desc_mean"] = pa.mean(0).round(4).tolist()
            ent["desc_std"] = pa.std(0).round(4).tolist()
        out["parts"][f"p{j}"] = ent
    if P > 1:
        d = []
        for a in range(P):
            for b in range(a + 1, P):
                x, y = m[:, K + 1 + a], m[:, K + 1 + b]
                inter = (x * y).reshape(len(x), -1).sum(1)
                d.append(np.mean(2 * inter / (x.reshape(len(x), -1).sum(1) + y.reshape(len(y), -1).sum(1) + eps)))
        out["mean_pairwise_soft_dice"] = float(np.mean(d))
    out["mean_area_frac_pred_lesion"] = float(np.mean([v["area_frac_pred_lesion"] for v in out["parts"].values()]))
    out["mean_mass_inside_expert_structures"] = float(np.mean([v["mass_inside_expert_structures"]
                                                               for v in out["parts"].values()]))
    return out


# ============================================================================ head sanity checks / ladder
def _lr_bacc(Xtr, ytr, Xva, yva, Xte, yte, Cs=(0.01, 0.1, 1.0, 10.0)) -> dict:
    """Class-balanced logistic regression on standardised inputs, C chosen on val BACC; returns test BACC."""
    import warnings
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    A, B, Cm = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for C in Cs:
            lr = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(A, ytr)
            bv = M.balanced_accuracy(yva, lr.predict(B), int(max(ytr.max(), yva.max(), yte.max()) + 1))
            if best is None or bv > best[0] + 1e-12:
                best = (bv, C, lr)
    nC = int(max(ytr.max(), yte.max()) + 1)
    return {"test_bacc": M.balanced_accuracy(yte, best[2].predict(Cm), nC), "val_bacc": float(best[0]),
            "C": best[1], "dim": int(Xtr.shape[1])}


@torch.no_grad()
def _split_pass(model, ds, bs=32) -> dict:
    dl = make_loader(ds, bs, False, 8)
    acc = {"image_id": [], "y_true": [], "c_prob": [], "c_true": [], "area_true": [], "head_inputs": []}
    for b in dl:
        x = b["image"].to("cuda", non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(x)
        acc["image_id"] += list(b["image_id"])
        acc["y_true"].append(b["label"].numpy())
        acc["c_prob"].append(torch.sigmoid(o["concept_logits"].float()).cpu().numpy())
        acc["c_true"].append(b["concepts"].numpy())
        acc["area_true"].append(b["area_true"].numpy())
        ex = o.get("extra", {}) or {}
        if "head_inputs" in ex:
            acc["head_inputs"].append(ex["head_inputs"].float().cpu().numpy())
    return {k: (np.array(v) if k == "image_id" else np.concatenate(v)) for k, v in acc.items() if len(v)}


def split_arrays(run_dir: Path, split: str, cfg: dict, model=None) -> dict | None:
    """y / concept probs / GT concepts+areas / head inputs for a split: from preds_<split>.npz if it has head
    inputs, else from a cached head_inputs_<split>.npz, else computed with the model (and cached)."""
    keys = ("image_id", "y_true", "c_prob", "c_true", "area_true", "head_inputs")
    p = run_dir / f"preds_{split}.npz"
    if p.exists():
        d = np.load(p, allow_pickle=True)
        if "head_inputs" in d.files or model is None:
            return {k: d[k] for k in keys if k in d.files}
    cache_p = run_dir / f"head_inputs_{split}.npz"
    if cache_p.exists():
        d = np.load(cache_p, allow_pickle=True)
        if "head_inputs" in d.files or model is None:
            return {k: d[k] for k in d.files}
    if model is None:
        return None
    ds = DermDataset(_dc(cfg)["dataset"], split, subset=_dc(cfg).get("subset" if split == "train" else "subset_eval", 0),
                     **dataset_kwargs(cfg))
    d = _split_pass(model, ds, (cfg.get("train") or {}).get("batch_size", 32))
    if "head_inputs" in d:          # never cache an incomplete pass (older in-memory model code)
        np.savez_compressed(cache_p, **d)
    return d


def _lesion_area_gt(cfg: dict, ids) -> np.ndarray:
    from cbm.data.datasets import load_manifest
    df = load_manifest(_dc(cfg)["dataset"]).set_index("image_id")
    return df.loc[list(ids), "lesion_area_img"].to_numpy(np.float32)


def head_sanity(run_dir: Path, cfg: dict, model=None, n_seeds: int = 5) -> dict:
    sp = {s: split_arrays(run_dir, s, cfg, model) for s in ("train", "val", "test")}
    if any(v is None for v in sp.values()):
        return {"available": False, "reason": "train-split inputs need the model (run without --fast)"}
    tr, va, te = sp["train"], sp["val"], sp["test"]
    y = (tr["y_true"], va["y_true"], te["y_true"])
    K = tr["c_true"].shape[1]
    C = int(max(y[0].max(), y[2].max()) + 1)

    seeds = [n_seeds]

    def fit(Xs, ytr=None):
        return _lr_bacc(Xs[0], y[0] if ytr is None else ytr, Xs[1], y[1], Xs[2], y[2])

    def rep(fn):
        vals = [fn(sd) for sd in range(seeds[0])]
        return {"mean": float(np.mean([v["test_bacc"] for v in vals])),
                "std": float(np.std([v["test_bacc"] for v in vals])), "dim": vals[0]["dim"], "seeds": len(vals)}
    out = {"available": True, "chance_bacc": 1.0 / C, "n_train": int(len(y[0])), "protocol":
           "logistic regression (class-balanced, standardised inputs, C in {0.01,0.1,1,10} chosen on val BACC) "
           "fit on the train split (eval transform), test BACC; random/shuffled averaged over seeds"}
    dt = np.load(run_dir / "preds_test.npz", allow_pickle=True) if (run_dir / "preds_test.npz").exists() else None
    if dt is not None:
        out["model_head_test_bacc"] = M.balanced_accuracy(dt["y_true"], dt["y_prob"].argmax(1), C)
    has_c = not np.isnan(tr["c_prob"]).all()
    has_h = all("head_inputs" in d for d in (tr, va, te))
    if has_h:
        H = (tr["head_inputs"], va["head_inputs"], te["head_inputs"])
        out["own_head_inputs"] = fit(H)
        D = H[0].shape[1]
        if D > 256:            # e.g. the black-box head (768-d): fewer repeats keep the CPU cost bounded
            seeds[0] = min(n_seeds, 3)
            out["seeds_note"] = "3 seeds for random/shuffled (head input dim > 256)"
        out["random_gaussian_full_dim"] = rep(lambda sd: fit(tuple(np.random.RandomState(sd * 3 + i).randn(len(h), D)
                                                                   for i, h in enumerate(H))))
        out["shuffled_labels_head_inputs"] = rep(lambda sd: fit(H, np.random.RandomState(100 + sd).permutation(y[0])))
    else:
        out["head_inputs_missing"] = "model has no extra['head_inputs'] (trained with older code): rerun evaluate_run"
    if has_c:
        P = (tr["c_prob"], va["c_prob"], te["c_prob"])
        out["own_concepts"] = fit(P)
        out["shuffled_labels_concepts"] = rep(lambda sd: fit(P, np.random.RandomState(200 + sd).permutation(y[0])))
    out["random_gaussian_K"] = rep(lambda sd: fit(tuple(np.random.RandomState(sd * 3 + i + 50).randn(len(d["y_true"]), K)
                                                        for i, d in enumerate((tr, va, te)))))
    G = tuple(np.nan_to_num(d["c_true"], nan=0.0) for d in (tr, va, te))
    out["oracle_concepts"] = fit(G)
    GA = tuple(np.concatenate([g, np.sqrt(np.clip(np.nan_to_num(d["area_true"], nan=0.0), 0, 1)),
                               _lesion_area_gt(cfg, d["image_id"])[:, None]], 1) for g, d in zip(G, (tr, va, te)))
    out["oracle_concepts_areas"] = fit(GA)
    return out


GROUP_ORDER = ["concept", "area", "report", "appearance", "lesion", "part", "verify", "cem_emb", "global"]


@torch.no_grad()
def head_ladder(run_dir: Path, cfg: dict, model=None) -> dict:
    sp = {s: split_arrays(run_dir, s, cfg, model) for s in ("train", "val", "test")}
    if any(v is None or "head_inputs" not in v for v in sp.values()):
        return {"available": False, "reason": "no head_inputs (older model code or --fast without dumps)"}
    names = list(getattr(model, "head_input_names", []) or [])
    if not names:
        dt = np.load(run_dir / "preds_test.npz", allow_pickle=True)
        names = list(dt["head_input_names"]) if "head_input_names" in dt.files else []
    tr, va, te = sp["train"], sp["val"], sp["test"]
    if len(names) != tr["head_inputs"].shape[1]:
        return {"available": False, "reason": "head_input_names do not match head_inputs"}
    groups = np.array([n.split(":")[0] for n in names])
    present = [g for g in GROUP_ORDER if g in set(groups)] + sorted(set(groups) - set(GROUP_ORDER))
    C = int(max(tr["y_true"].max(), te["y_true"].max()) + 1)
    can_head = model is not None and hasattr(model, "head_forward")
    mu = tr["head_inputs"].mean(0)
    steps, cur = [], []
    for g in present:
        cur = cur + [g]
        cols = np.isin(groups, cur)
        ent = {"groups": list(cur), "n_inputs": int(cols.sum()),
               "refit": _lr_bacc(tr["head_inputs"][:, cols], tr["y_true"], va["head_inputs"][:, cols], va["y_true"],
                                 te["head_inputs"][:, cols], te["y_true"])["test_bacc"]}
        if can_head:
            for mode in ("zeroed", "mean_imputed"):
                X = te["head_inputs"].copy()
                X[:, ~cols] = 0.0 if mode == "zeroed" else mu[~cols]
                lg = model.head_forward(torch.from_numpy(X).to(next(model.parameters()).device)).float().cpu().numpy()
                ent[f"head_{mode}"] = M.balanced_accuracy(te["y_true"], lg.argmax(1), C)
        steps.append(ent)
    single = {}
    for g in present:     # label information carried by each input group on its own (non-concept channels = leakage)
        cols = groups == g
        single[g] = _lr_bacc(tr["head_inputs"][:, cols], tr["y_true"], va["head_inputs"][:, cols], va["y_true"],
                             te["head_inputs"][:, cols], te["y_true"])["test_bacc"]
    nonc = ~np.isin(groups, ["concept"])
    if nonc.any() and (~nonc).any():
        single["all_but_concept"] = _lr_bacc(tr["head_inputs"][:, nonc], tr["y_true"], va["head_inputs"][:, nonc],
                                             va["y_true"], te["head_inputs"][:, nonc], te["y_true"])["test_bacc"]
    return {"available": True, "groups_present": present, "steps": steps, "single_group_refit": single,
            "group_sizes": {g: int((groups == g).sum()) for g in present},
            "note": "refit = logistic regression on the subset (train->test, C on val); head_zeroed / "
                    "head_mean_imputed = the trained head with the other inputs set to 0 / their train mean"}


# ============================================================================ figures
PART_COL = ["#4a3aa7", "#e34948", "#008300", "#eda100", "#1baf7a", "#e87ba4"]


def part_examples_figure(out_dir: Path, cache, base, n_ex: int = 6):
    """For each discovered part: the n_ex test crops whose part descriptor is nearest to the part's mean descriptor
    (among images where the part is present), shown as the soft-mask-weighted crop. For naming the parts."""
    import cv2
    import matplotlib.pyplot as plt
    K = cache.K
    m = base["masks"]
    P = m.shape[1] - K - 1
    fig, axes = plt.subplots(P, n_ex, figsize=(1.3 * n_ex, 1.4 * P), dpi=150, squeeze=False)
    for j in range(P):
        pm = m[:, K + 1 + j]
        present = np.flatnonzero((pm >= 0.5).reshape(len(pm), -1).mean(1) >= 0.002)
        if len(present) == 0:
            for a in axes[j]:
                a.axis("off")
            axes[j, 0].set_title(f"part {j}: never present", fontsize=6, loc="left")
            continue
        desc = base["part_app"][present, j]
        order = present[np.argsort(np.linalg.norm(desc - desc.mean(0), axis=1))[:n_ex]]
        for c, i in enumerate(order):
            bi = next(b for b, (s, e) in enumerate(cache.slices) if s <= i < e)
            img = denormalize(cache.x[bi][i - cache.slices[bi][0]].float()).permute(1, 2, 0).cpu().numpy()
            H, W = img.shape[:2]
            soft = cv2.resize(pm[i].astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
            ys, xs = np.nonzero(soft >= 0.5)
            if len(ys) == 0:
                ys, xs = np.nonzero(soft >= soft.max() * 0.5)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            side = int(max(y1 - y0, x1 - x0) * 1.2) + 4
            cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
            ya, xa = max(0, cy - side // 2), max(0, cx - side // 2)
            crop = (img * np.clip(soft, 0, 1)[..., None])[ya:ya + side, xa:xa + side]
            ax = axes[j, c]
            ax.imshow(cv2.resize(crop, (96, 96))); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if c == 0:
                ax.set_ylabel(f"part {j}", fontsize=7)
        for c in range(len(order), n_ex):
            axes[j, c].axis("off")
    fig.tight_layout()
    p = out_dir / "parts_examples.png"
    fig.savefig(p); plt.close(fig)
    return p


def make_figures(run_id: str, ev: dict, cache=None, base=None, examples=None) -> list[tuple[str, str]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out_dir = FIG_ROOT / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#8a8984",
                         "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                         "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
    figs = []
    mt = ev.get("model_tests", {})
    if "tti" in mt:
        fig, ax = plt.subplots(figsize=(4.6, 3.2), dpi=150)
        cols = {"random": SER[0], "uncertainty": SER[1], "verification": SER[2]}
        for name, cur in mt["tti"].items():
            base_n = name.split("_noisy")[0]
            ls = "--" if "_noisy" in name else "-"
            ax.plot(cur["k"], cur["bacc"], ls, color=cols.get(base_n, SER[3]), lw=2, marker="o", ms=4,
                    label=name.replace("_noisy80", " (expert p=0.8)"))
        ax.set_xlabel("concepts intervened (k)"); ax.set_ylabel("balanced accuracy")
        ax.set_title(f"Test-time interventions - {run_id}", fontsize=9, loc="left")
        ax.legend(fontsize=7)
        p = out_dir / "tti.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig)
        figs.append((str(p), f"{run_id}: TTI curves"))
    if "spatial" in mt:
        fig, ax = plt.subplots(figsize=(4.6, 3.2), dpi=150)
        for j, name in enumerate(["random", "uncertainty"]):
            if name in mt["spatial"]:
                cur = mt["spatial"][name]
                ax.plot(cur["k"], cur["bacc"], "-", color=SER[j], lw=2, marker="o", ms=4, label=f"{name} order")
        ax.set_xlabel("predicted masks replaced by expert masks (k)"); ax.set_ylabel("balanced accuracy")
        ax.set_title(f"Spatial interventions - {run_id}", fontsize=9, loc="left")
        ax.legend(fontsize=7)
        p = out_dir / "spatial.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig)
        figs.append((str(p), f"{run_id}: spatial interventions"))
    dfr = mt.get("deferral") or ev.get("test", {}).get("deferral")
    if dfr:
        fig, ax = plt.subplots(figsize=(4.6, 3.2), dpi=150)
        for j, (name, cur) in enumerate(dfr.items()):
            ax.plot(cur["curve_cov"], cur["curve_acc"], color=SER[j % 8], lw=2, label=f"{name} (AUC {cur['auc']:.3f})")
        ax.set_xlabel("coverage"); ax.set_ylabel("accuracy of retained cases")
        ax.set_title(f"Deferral - {run_id}", fontsize=9, loc="left")
        ax.legend(fontsize=7)
        p = out_dir / "deferral.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig)
        figs.append((str(p), f"{run_id}: accuracy-coverage"))
    cf = mt.get("counterfactual", {})
    if cf.get("available") and cf.get("per_concept"):
        names = list(cf["per_concept"])
        fig, ax = plt.subplots(figsize=(4.6, 3.0), dpi=150)
        xx = np.arange(len(names))
        ax.bar(xx - 0.2, [cf["per_concept"][n]["CLS"] for n in names], 0.38, color=SER[0], label="CLS (higher better)")
        ax.bar(xx + 0.2, [cf["per_concept"][n]["XE"] for n in names], 0.38, color=SER[1], label="XE (lower better)")
        ax.set_xticks(xx); ax.set_xticklabels([n.replace("_", " ") for n in names], rotation=20, fontsize=7)
        ax.axhline(0, color="#8a8984", lw=0.8)
        ax.set_title(f"Counterfactual locality - {run_id}", fontsize=9, loc="left")
        ax.legend(fontsize=7)
        p = out_dir / "counterfactual.png"; fig.tight_layout(); fig.savefig(p); plt.close(fig)
        figs.append((str(p), f"{run_id}: CLS / XE per concept"))
    if cache is not None and base is not None and "masks" in base and len(base["masks"]):
        import cv2
        names = cache_concepts(cache)
        idx = [i for i in np.argsort(-np.nansum(cache.c, 1))[:4]]
        fig, axes = plt.subplots(len(idx), 3, figsize=(6.6, 2.2 * len(idx)), dpi=150)
        for r, i in enumerate(idx):
            bi = next(b for b, (s, e) in enumerate(cache.slices) if s <= i < e)
            s0 = cache.slices[bi][0]
            img = (denormalize(cache.x[bi][i - s0].float()).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img = cv2.resize(img, (256, 256))
            gt = (cache.masks[i].numpy() >= 128)
            pr = base["masks"][i] >= 0.5
            panels = [img.copy(), img.copy(), img.copy()]
            for k in range(cache.K + 1):
                col = tuple(int(SER[k % 8][j:j + 2], 16) for j in (1, 3, 5)) if k < cache.K else (255, 255, 255)
                for pi, m in [(1, gt[k]), (2, pr[k])]:
                    mm = cv2.resize(m.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
                    cs, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                    cv2.drawContours(panels[pi], cs, -1, col, 2 if k < cache.K else 1)
            for jp in range(pr.shape[0] - cache.K - 1):           # discovered parts (predicted panel only)
                col = tuple(int(PART_COL[jp % len(PART_COL)][j:j + 2], 16) for j in (1, 3, 5))
                mm = cv2.resize(pr[cache.K + 1 + jp].astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST)
                cs, _ = cv2.findContours(mm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(panels[2], cs, -1, col, 1, lineType=cv2.LINE_4)
            for c_, (pan, t) in enumerate(zip(panels, ["image", "expert masks", "predicted masks"])):
                ax = axes[r, c_]; ax.imshow(pan); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
                if r == 0:
                    ax.set_title(t, fontsize=8)
        handles = [plt.Line2D([], [], color=SER[k], lw=2, label=names[k].replace("_", " ")) for k in range(cache.K)]
        handles.append(plt.Line2D([], [], color="#8a8984", lw=1, label="lesion"))
        n_parts_fig = base["masks"].shape[1] - cache.K - 1
        handles += [plt.Line2D([], [], color=PART_COL[j % len(PART_COL)], lw=1, label=f"part {j}")
                    for j in range(n_parts_fig)]
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        p = out_dir / "masks_overlay.png"; fig.savefig(p); plt.close(fig)
        figs.append((str(p), f"{run_id}: expert vs predicted masks"))
        if n_parts_fig > 0 and "part_app" in base:
            p = part_examples_figure(out_dir, cache, base, n_ex=6)
            if p:
                figs.append((str(p), f"{run_id}: discovered parts - typical crops (nearest to mean descriptor)"))
    if examples:
        fig, axes = plt.subplots(1, len(examples), figsize=(2.0 * len(examples), 2.2), dpi=150)
        for ax, (k, im) in zip(np.atleast_1d(axes), examples.items()):
            ax.imshow(im.permute(1, 2, 0).numpy()); ax.set_title(k, fontsize=8); ax.axis("off")
        fig.tight_layout()
        p = out_dir / "artefact_examples.png"; fig.savefig(p); plt.close(fig)
        figs.append((str(p), f"{run_id}: synthetic artefacts (outside lesion)"))
    return figs


# ============================================================================ main
def summarize(ev: dict) -> dict:
    s = {}
    t = ev.get("test", {})
    for k in ("acc", "bacc", "macro_f1", "auroc_macro", "auroc_mel", "ece"):
        s[f"test_{k}"] = t.get("task", {}).get(k)
    s["test_concept_auroc"] = t.get("concepts", {}).get("mean_auroc")
    g = t.get("grounding", {})
    s["test_dice"] = g.get("mean_dice"); s["test_iou"] = g.get("mean_iou"); s["test_pointing"] = g.get("mean_pointing")
    s["test_lesion_dice"] = g.get("lesion", {}).get("dice")
    s["ois"] = t.get("ois", {}).get("ois")
    s["leak_soft_minus_hard_cv"] = t.get("leakage_probe", {}).get("soft_minus_hard")
    mt = ev.get("model_tests", {})
    h = mt.get("havasi", {})
    s["havasi_bacc_gt"] = h.get("bacc_gt"); s["havasi_soft_minus_hard"] = h.get("soft_minus_hard")
    s["havasi_intervention_gain"] = h.get("intervention_gain")
    for pol in ("random", "uncertainty", "verification", "random_noisy80"):
        if pol in mt.get("tti", {}):
            s[f"tti_area_{pol}"] = mt["tti"][pol]["area_bacc"]
    if "spatial" in mt:
        s["spatial_area_random"] = mt["spatial"]["random"]["area_bacc"]
    loc = mt.get("locality", {})
    if loc:
        s["locality_gap_blur"] = loc["blur"]["mean_gap_in_minus_out"]
        s["locality_ratio_blur"] = loc["blur"]["mean_ratio_out_in"]
        s["task_tv_outside_lesion_blur"] = loc["blur"]["task_tv_outside_lesion"]
    cf = mt.get("counterfactual", {})
    s["CLS"] = cf.get("mean_CLS"); s["XE"] = cf.get("mean_XE")
    art = mt.get("artefacts", {})
    s["artefact_flip_rate"] = art.get("mean_flip_rate"); s["artefact_delta_bacc"] = art.get("mean_delta_bacc")
    dfr = mt.get("deferral", t.get("deferral", {}))
    for k in ("maxprob", "verification"):
        if k in dfr:
            s[f"deferral_auc_{k}"] = dfr[k]["auc"]
    cal = ev.get("calibration", {}) or {}
    if cal.get("available"):
        s["test_ece_ts"] = cal["test_ece_ts"]; s["test_nll_ts"] = cal["test_nll_ts"]; s["temperature"] = cal["temperature"]
    ph2 = ev.get("ph2", {})
    s["ph2_auroc"] = ph2.get("auroc_mel_vs_nevus"); s["ph2_bacc"] = ph2.get("bacc_mel_vs_nevus")
    san = ev.get("sanity", {}) or {}
    if san.get("available"):
        for k in ("own_head_inputs", "own_concepts", "oracle_concepts", "oracle_concepts_areas"):
            if k in san:
                s[f"sanity_{k}"] = san[k]["test_bacc"]
        for k in ("random_gaussian_full_dim", "shuffled_labels_head_inputs", "random_gaussian_K",
                  "shuffled_labels_concepts"):
            if k in san:
                s[f"sanity_{k}"] = san[k]["mean"]
    lad = ev.get("ladder", {}) or {}
    if lad.get("available"):
        for st in lad["steps"]:
            s["ladder_refit_" + "+".join(st["groups"])] = st["refit"]
    loc = mt.get("locality", {}).get("blur", {})
    if "lesion_emb" in loc:
        s["lesion_emb_drift_out_in_ratio_blur"] = loc["lesion_emb"]["ratio_out_in"]
    fxl, fxs = loc.get("fixed_masks_outside_pred_lesion", {}), loc.get("fixed_masks_outside_pred_struct", {})
    if fxl:
        s["fixedmask_task_tv_outside_lesion_blur"] = fxl["task_tv"]
        if "lesion_emb_rel_drift" in fxl:
            s["fixedmask_lesion_emb_drift_blur"] = fxl["lesion_emb_rel_drift"]
    if fxs:
        s["fixedmask_concept_delta_outside_struct_blur"] = fxs["concept_abs_delta"]
    fxp = loc.get("fixed_masks_outside_part_support", {})
    if fxp:
        s["fixedmask_part_desc_maxdelta_blur"] = fxp["part_desc_max_abs_delta"]
    pr = mt.get("parts", {})
    if pr.get("available"):
        s["parts_mean_area_frac_lesion"] = pr["mean_area_frac_pred_lesion"]
        s["parts_mass_inside_structures"] = pr["mean_mass_inside_expert_structures"]
    fxe = loc.get("fixed_expert_masks_outside_struct", {})
    if fxe:
        s["fixed_expert_mask_concept_logit_maxdelta_blur"] = fxe["concept_logit_max_abs_delta"]
    return {k: v for k, v in s.items() if v is not None}


ALL_TESTS = ["calibration", "havasi", "tti", "spatial", "deferral", "locality", "counterfactual", "artefacts", "parts",
             "sanity", "ladder"]
DUMP_ONLY_TESTS = {"calibration", "dump"}      # need no model / GPU


def evaluate_run(run_id: str, model=None, full: bool = True, device: str = "cuda", tests: list[str] | None = None,
                 merge: bool = False):
    import yaml
    run_dir = T.RUNS_DIR / run_id
    t0 = time.time()
    torch.backends.cudnn.benchmark = False     # many differently-sized batches below
    cfg = yaml.safe_load(open(run_dir / "config.yaml"))
    log = lambda *m: print(f"[eval {run_id}]", *m, flush=True)
    if merge and (run_dir / "eval.json").exists():
        ev = json.load(open(run_dir / "eval.json"))
        ev["time_merged"] = time.time()
    else:
        merge = False
        ev = {"run_id": run_id, "config_file": cfg.get("_config_file"), "time": time.time()}
    if not merge or (tests and "dump" in tests):
        for split in ("val", "test"):
            ev[split] = dump_eval(run_dir, split, cfg)
            log(f"dump metrics {split} done ({time.time() - t0:.0f}s)")
        ev["ph2"] = ph2_metrics(run_dir)
    figs_args = {}
    todo_all = list(tests or ALL_TESTS)
    if "calibration" in todo_all:
        try:
            ev["calibration"] = temperature_scaling(run_dir)
            log(f"calibration done: T={ev['calibration'].get('temperature')}")
        except Exception as e:
            ev["calibration"] = {"available": False, "error": str(e)}
    need_model = any(t not in DUMP_ONLY_TESTS for t in todo_all)
    if full and need_model:
        if model is None:
            from cbm.models import load_model
            model, cfg_m, _ = load_model(run_dir, device)
        model.eval()
        if "ph2" in todo_all:
            try:
                ph2_dump(run_dir, cfg, model, device)
                ev["ph2"] = ph2_metrics(run_dir)
                log(f"ph2 done: AUROC {ev['ph2'].get('auroc_mel_vs_nevus')} (n={ev['ph2'].get('n')})")
            except Exception as e:
                import traceback
                ev["ph2"] = {"available": False, "error": str(e), "trace": traceback.format_exc()[-1500:]}
                log(f"ph2 FAILED: {e}")
    if full and need_model and any(t not in DUMP_ONLY_TESTS | {"ph2"} for t in todo_all):
        ds = DermDataset(_dc(cfg)["dataset"], "test", subset=_dc(cfg).get("subset_eval", 0), **dataset_kwargs(cfg))
        cache = Cache(model, ds, device, (cfg.get("train") or {}).get("batch_size", 32), _dc(cfg).get("num_workers", 8))
        cache.concept_names, cache.class_names = ds.concept_names, ds.class_names
        base = cache.run(model, keep_masks=True)
        log(f"cache built N={cache.N} heads_cached={cache.use_heads} ({time.time() - t0:.0f}s)")
        has_c = not np.isnan(base["c_prob"]).all()
        has_m = "masks" in base and len(base["masks"]) > 0
        mt = ev.get("model_tests", {}) if merge else {}
        mt["base"] = _summ(cache.y, base, cache.c)
        todo = [t for t in todo_all if t not in DUMP_ONLY_TESTS | {"ph2"}]
        examples = None
        for name in todo:
            ts = time.time()
            try:
                if name == "havasi" and has_c:
                    mt["havasi"] = havasi(cache, model, base)
                elif name == "tti" and has_c:
                    mt["tti"] = tti(cache, model, base)
                elif name == "spatial" and has_m and has_c:
                    mt["spatial"] = spatial(cache, model, base)
                elif name == "deferral":
                    mt["deferral"] = deferral(cache, model, base)
                elif name == "locality":
                    mt["locality"] = locality(cache, model, base, _dc(cfg)["image_size"])
                elif name == "counterfactual":
                    mt["counterfactual"] = counterfactual(cache, model, base, _dc(cfg)["image_size"])
                elif name == "artefacts":
                    mt["artefacts"], examples = artefacts(cache, model, base, _dc(cfg)["image_size"])
                elif name == "parts":
                    mt["parts"] = parts_report(cache, model, base)
                elif name == "sanity":
                    ev["sanity"] = head_sanity(run_dir, cfg, model)
                elif name == "ladder":
                    ev["ladder"] = head_ladder(run_dir, cfg, model)
                else:
                    continue
                log(f"{name} done ({time.time() - ts:.0f}s)")
            except Exception as e:
                import traceback
                err = {"error": str(e), "trace": traceback.format_exc()[-2000:]}
                if name in ("sanity", "ladder"):
                    ev[name] = err
                else:
                    mt[name] = err
                log(f"{name} FAILED: {e}")
        mt["report_weights"] = report_weights(model, cfg)
        ev["model_tests"] = mt
        figs_args = dict(cache=cache, base=base, examples=examples)
    if not full and tests:          # --fast: sanity / ladder from dumps + cached head inputs only
        for name in ("sanity", "ladder"):
            if name in tests:
                ev[name] = (head_sanity if name == "sanity" else head_ladder)(run_dir, cfg, None)
    ev["summary"] = summarize(ev)
    ev["eval_seconds"] = time.time() - t0
    with open(run_dir / "eval.json", "w") as f:
        json.dump(_clean(ev), f, indent=1)
    if figs_args or not merge:      # a calibration-only merge does not touch the figures
        try:
            for path, title in make_figures(run_id, ev, **figs_args):
                T.add_figure(path, title, group=run_id)
        except Exception as e:
            log(f"figures failed: {e}")
    import csv
    with open(run_dir / "eval_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id"] + list(ev["summary"]))
        w.writerow([run_id] + [ev["summary"][k] for k in ev["summary"]])
    log(f"wrote eval.json ({time.time() - t0:.0f}s): {json.dumps(_clean(ev['summary']))}")
    return ev


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--fast", action="store_true", help="dump-based metrics only (no model reload)")
    ap.add_argument("--tests", nargs="*", default=None, help=f"subset of {ALL_TESTS} (+ 'dump', 'ph2')")
    ap.add_argument("--merge", action="store_true", help="update an existing eval.json with the requested tests")
    a = ap.parse_args()
    if a.gpu is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    evaluate_run(a.run_id, full=not a.fast, tests=a.tests, merge=a.merge)
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
