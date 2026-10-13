"""One trainer for every EB-CBM config (flagship and ablations).

    python src/cbm/scripts/train.py --config configs/ebcbm.yaml --run_id ebcbm_s0 --gpu 0 [--set seed=1 ...]

AdamW (backbone / head LRs), linear warmup + cosine, bf16 autocast, grad clipping, class-weighted CE or
class-balanced sampler, early stopping on val BACC, best.pt, tracker integration (register_run,
log_metrics train/val each epoch, update_run status/best), prediction dumps preds_{val,test[,ph2]}.npz in the
contract format, then (optionally) cbm.eval.evaluate_run.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from cbm import track as T
from cbm.datamod import DermDataset, PH2Dataset, collate, dataset_kwargs, make_loader
from cbm.models import build_model
from cbm.models.heads import dice_score, mask_loss, sample_verifier_regions
from cbm.paths import CF_INDEX

RUNS = T.RUNS_DIR


# ============================================================================ utilities
class RunLogger:
    def __init__(self, path: Path):
        self.path = path
        # when launched by scripts/queue.py stdout is already redirected to log.txt
        self.to_file = os.environ.get("CBM_STDOUT_IS_LOG", "0") != "1"

    def __call__(self, *msg):
        s = f"[{time.strftime('%H:%M:%S')}] " + " ".join(str(m) for m in msg)
        print(s, flush=True)
        if self.to_file:
            with open(self.path, "a") as f:
                f.write(s + "\n")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def slice_out(out: dict, sl) -> dict:
    def s(v):
        return v[sl] if isinstance(v, torch.Tensor) and v.dim() > 0 else v
    return {"logits": s(out["logits"]), "concept_logits": s(out["concept_logits"]),
            "masks": s(out["masks"]) if out.get("masks") is not None else None,
            "extra": {k: s(v) for k, v in out.get("extra", {}).items()}}


def task_metrics(y: np.ndarray, prob: np.ndarray) -> dict:
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    pred = prob.argmax(1)
    m = {"acc": float((pred == y).mean()), "bacc": float(balanced_accuracy_score(y, pred))}
    try:
        present = np.unique(y)
        if len(present) == prob.shape[1]:
            m["auc"] = float(roc_auc_score(y, prob, multi_class="ovr", average="macro"))
        else:
            m["auc"] = float(np.mean([roc_auc_score(y == k, prob[:, k]) for k in present]))
        if 1 in present and len(present) > 1:
            m["auc_mel"] = float(roc_auc_score(y == 1, prob[:, 1]))
    except ValueError:
        pass
    return m


def concept_auc(c: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    aucs = []
    for k in range(c.shape[1]):
        ok = ~np.isnan(c[:, k]) & ~np.isnan(p[:, k])
        if ok.sum() > 1 and len(np.unique(c[ok, k])) == 2:
            aucs.append(roc_auc_score(c[ok, k], p[ok, k]))
    return float(np.mean(aucs)) if aucs else float("nan")


def cosine_warmup(step: int, total: int, warm: int) -> float:
    if step < warm:
        return (step + 1) / max(1, warm)
    q = (step - warm) / max(1, total - warm)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, q)))


# ============================================================================ prediction
@torch.no_grad()
def predict(model, loader, device="cuda", bf16=True, mask_out: int = 64, keep_features: bool = False) -> dict:
    """Run the model on a loader and return the contract dump (+ extras)."""
    model.eval()
    acc: dict[str, list] = {k: [] for k in ["image_id", "y_true", "y_prob", "logits", "c_true", "c_prob",
                                            "area_true", "area_pred", "mask_pred", "verify", "c_repr", "report",
                                            "mask_true", "head_inputs", "part_desc", "part_area", "proto_sim"]}
    for b in loader:
        x = b["image"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            out = model(x)
        lg = out["logits"].float()
        acc["image_id"] += list(b["image_id"])
        acc["y_true"].append(b["label"].numpy())
        acc["logits"].append(lg.cpu().numpy())
        acc["y_prob"].append(F.softmax(lg, -1).cpu().numpy())
        acc["c_true"].append(b["concepts"].numpy())
        acc["c_prob"].append(torch.sigmoid(out["concept_logits"].float()).cpu().numpy())
        acc["area_true"].append(b["area_true"].numpy())
        ex = out.get("extra", {})
        if out.get("masks") is not None:
            m = out["masks"].float()
            m64 = F.adaptive_avg_pool2d(m, (mask_out, mask_out))
            acc["mask_pred"].append((m64 * 255 + 0.5).clamp(0, 255).byte().cpu().numpy())
            acc["area_pred"].append(ex["area_lesion"].float().cpu().numpy())
            gt = F.adaptive_avg_pool2d(b["masks"].float(), (mask_out, mask_out)) >= 0.5
            acc["mask_true"].append(gt.numpy())
        if "verify" in ex:
            acc["verify"].append(ex["verify"].float().cpu().numpy())
        if "concept_repr" in ex:
            acc["c_repr"].append(ex["concept_repr"].float().cpu().numpy())
        if "report" in ex:
            acc["report"].append(ex["report"].float().cpu().numpy())
        if "head_inputs" in ex:
            acc["head_inputs"].append(ex["head_inputs"].float().cpu().numpy())
        if "part_app" in ex:
            acc["part_desc"].append(ex["part_app"].float().cpu().numpy())
            acc["part_area"].append(ex["part_area"].float().cpu().numpy())
        if "proto_sim" in ex:
            acc["proto_sim"].append(ex["proto_sim"].float().cpu().numpy())
    out = {}
    for k, v in acc.items():
        if not v:
            continue
        out[k] = np.array(v) if k == "image_id" else np.concatenate(v)
    K = out["c_true"].shape[1]
    if "area_pred" not in out:
        out["area_pred"] = np.full((len(out["y_true"]), K), np.nan, np.float32)
    return out


def save_dump(path: Path, d: dict, class_names, concept_names, extra_names: dict | None = None) -> None:
    keep = {k: v for k, v in d.items() if k != "mask_true"}
    keep["class_names"] = np.array(class_names)
    keep["concept_names"] = np.array(concept_names)
    for k, v in (extra_names or {}).items():
        keep[k] = np.array(v)
    np.savez_compressed(path, **keep)


def dump_metrics(d: dict) -> dict:
    m = task_metrics(d["y_true"], d["y_prob"])
    m["concept_auc"] = concept_auc(d["c_true"], d["c_prob"])
    if "mask_pred" in d and "mask_true" in d:
        p = d["mask_pred"] >= 128
        t = d["mask_true"]
        K = t.shape[1] - 1                     # pred may carry extra (discovered part) planes after the lesion
        dices = []
        for k in range(K + 1):
            inter = (p[:, k] & t[:, k]).sum()
            den = p[:, k].sum() + t[:, k].sum()
            dices.append(2 * inter / den if den > 0 else np.nan)
        m["mask_dice"] = float(np.nanmean(dices[:K]))
        m["lesion_dice"] = float(dices[K])
    return m


# ============================================================================ trainer
class Trainer:
    def __init__(self, cfg: dict, run_id: str, gpu: int = 0):
        self.cfg, self.run_id, self.gpu = cfg, run_id, gpu
        self.dir = RUNS / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = RunLogger(self.dir / "log.txt")
        self.device = "cuda"
        tc, dc = cfg["train"], cfg["data"]
        seed_all(cfg.get("seed", 0))
        cap = tc.get("gpu_mem_gb")
        if cap:   # hard per-process cap (packing jobs / smoke tests next to running jobs)
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            torch.cuda.set_per_process_memory_fraction(min(1.0, float(cap) / total), 0)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # variable shapes (CF pairs per batch, verifier region counts) would re-trigger cuDNN autotuning every step
        torch.backends.cudnn.benchmark = False
        with open(self.dir / "config.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        # ---------------- data
        cf_on = bool(cfg.get("cf", {}).get("enabled", False))
        common = dataset_kwargs(cfg)
        self.ds_tr = DermDataset(dc["dataset"], "train", train=True, aug=dc.get("aug"), subset=dc.get("subset", 0),
                                 cf_index=(dc.get("cf_index") or str(CF_INDEX)) if cf_on else None,
                                 cf_frac=cfg["cf"]["frac"] if cf_on else 0.0, **common)
        se = dc.get("subset_eval", 0)
        self.ds_va = DermDataset(dc["dataset"], "val", subset=se, **common)
        self.ds_te = DermDataset(dc["dataset"], "test", subset=se, **common)
        self.class_names, self.concept_names = self.ds_tr.class_names, self.ds_tr.concept_names
        self.C, self.K = len(self.class_names), len(self.concept_names)
        nw = dc.get("num_workers", 8)
        sampler = None
        counts = np.bincount(self.ds_tr.labels, minlength=self.C).astype(np.float64)
        if tc.get("balance") == "sampler":
            w = (1.0 / counts)[self.ds_tr.labels]
            sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(w), len(w), replacement=True,
                                                             generator=torch.Generator().manual_seed(cfg.get("seed", 0)))
        self.class_w = None
        if tc.get("balance") == "loss":
            w = counts.sum() / (self.C * np.maximum(counts, 1))
            self.class_w = torch.tensor(w, dtype=torch.float32, device=self.device)
        self.dl_tr = make_loader(self.ds_tr, tc["batch_size"], True, nw, sampler, seed=cfg.get("seed", 0))
        self.dl_va = make_loader(self.ds_va, tc["batch_size"], False, nw)
        self.dl_te = make_loader(self.ds_te, tc["batch_size"], False, nw)
        cp = self.ds_tr.concepts
        pos = np.nansum(cp, 0)
        neg = np.sum(~np.isnan(cp), 0) - pos
        self.pos_weight = None
        if cfg["loss"].get("concept_pos_weight") == "auto":
            self.pos_weight = torch.tensor(np.clip(np.sqrt(neg / np.maximum(pos, 1)), 1, 10), dtype=torch.float32,
                                           device=self.device)
        # ---------------- model
        self.model = build_model(cfg, self.C, self.concept_names).to(self.device)
        if hasattr(self.model, "area_prior"):
            ai = self.ds_tr.df[[f"{c}_area_img" for c in self.concept_names]].to_numpy(np.float32)
            pr = [float(np.nanmean(ai[:, k][ai[:, k] > 0])) if (ai[:, k] > 0).any() else 0.02 for k in range(self.K)]
            self.model.area_prior.copy_(torch.tensor(pr))
        mcfg = cfg["model"]
        self.use_mask = bool(mcfg.get("mask_head", False))
        self.has_concepts = mcfg.get("concept_head", "none") != "none"
        self.use_verifier = bool((mcfg.get("verifier") or {}).get("enabled", False))
        self.is_report = mcfg.get("diag_head") == "report"
        self.cf_on = cf_on
        groups = self.model.param_groups(tc["lr_backbone"], tc["lr_heads"], tc["weight_decay"],
                                         lr_local=tc.get("lr_local", 1e-4), lr_c2y=tc.get("lr_c2y", 1e-2)) \
            if hasattr(self.model, "param_groups") else self.model.parameters()
        self.opt = torch.optim.AdamW(groups, betas=(0.9, 0.999))
        self.epochs = int(tc["epochs"])
        self.steps_per_epoch = len(self.dl_tr)
        total = self.epochs * self.steps_per_epoch
        warm = int(tc.get("warmup_epochs", 1) * self.steps_per_epoch)
        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lambda s: cosine_warmup(s, total, warm))
        self.lr_groups = {g.get("name", str(i)): (g["lr"], sum(p.numel() for p in g["params"]))
                          for i, g in enumerate(self.opt.param_groups)}
        n_par = sum(p.numel() for p in self.model.parameters())
        self.log(f"run {run_id}: train={len(self.ds_tr)} val={len(self.ds_va)} test={len(self.ds_te)} "
                 f"params={n_par/1e6:.1f}M steps/epoch={self.steps_per_epoch} class_w={self.class_w} "
                 f"pos_w={self.pos_weight} cf={cf_on} lr_groups(lr, n_params)={self.lr_groups}")

    # ------------------------------------------------------------------ losses
    def compute_loss(self, out, b, B, X):
        lc = self.cfg["loss"]
        tc = self.cfg["train"]
        logs = {}
        o = slice_out(out, slice(0, B)) if out["logits"].shape[0] != B else out
        y = b["label"]
        l_task = F.cross_entropy(o["logits"].float(), y, weight=self.class_w,
                                 label_smoothing=tc.get("label_smoothing", 0.0))
        total = lc["task"] * l_task
        logs["loss_task"] = float(l_task.detach())
        if self.has_concepts:
            c, kn = b["concepts"], b["concept_known"].float()
            l = F.binary_cross_entropy_with_logits(o["concept_logits"].float(), torch.nan_to_num(c, 0.0),
                                                   reduction="none", pos_weight=self.pos_weight)
            l_c = (l * kn).sum() / kn.sum().clamp_min(1)
            total = total + lc["concept"] * l_c
            logs["loss_concept"] = float(l_c.detach())
        if self.use_mask:
            l_m, ml = mask_loss(o["extra"]["mask_logits"], b["masks"], b["mask_valid"])
            total = total + lc["mask"] * l_m
            logs["loss_mask"] = float(l_m.detach())
        if hasattr(self.model, "prototypes") and getattr(self.model, "proto_div", 0) > 0:
            l_div = self.model.proto_diversity()
            total = total + self.model.proto_div * l_div
            logs["loss_proto_div"] = float(l_div.detach())
        n_parts = getattr(self.model, "n_parts", 0)
        if n_parts > 0:
            from cbm.models.heads import part_regularizers
            K = self.K
            dc = self.cfg["model"].get("discover", {}) or {}
            rw = {"outside": 1.0, "compact": 0.1, "overlap": 1.0, "area": 1.0, "binary": 0.0, "peak": 0.0,
                  **(dc.get("reg") or {})}
            Mp = o["masks"][:, K + 1:K + 1 + n_parts]
            lv = b["mask_valid"][:, K][:, None, None]
            L = torch.where(lv, b["masks"][:, K], o["masks"][:, K].detach())          # expert lesion where available
            Sx = b["masks"][:, :K] * b["mask_valid"][:, :K, None, None].float()
            regs = part_regularizers(Mp, L, Sx, float(rw.get("area_min", 0.01)), float(rw.get("area_max", 0.30)))
            for k, v in regs.items():
                total = total + float(rw.get(k, 0.0)) * v
                logs[f"part_{k}"] = float(v.detach())
        if self.is_report:
            l1 = self.cfg["model"].get("report", {}).get("l1", 0.0)
            if l1 > 0:
                total = total + l1 * self.model.diag.l1()
        if self.cf_on and out["logits"].shape[0] > B:
            from cbm.train.counterfactual import cf_loss
            has = b["cf_has"]
            nb = int(has.sum())
            ox = slice_out(o, has)
            om = slice_out(out, slice(B, B + nb))
            oc = slice_out(out, slice(B + nb, B + 2 * nb))
            l_cf, cl = cf_loss(ox, om, oc, b["concepts"][has], b["cf_concepts_minus"][has], b["concept_known"][has],
                               b["masks"][has], b["cf_masks_minus"][has], b["mask_valid"][has], self.has_concepts,
                               self.use_mask, lc.get("cf_kl", 1.0), self.pos_weight)
            total = total + lc["cf"] * l_cf
            logs["loss_cf"] = float(l_cf.detach())
            logs.update(cl)
        if self.use_verifier:
            R, idx, t, known = sample_verifier_regions(b["masks"], b["mask_valid"], o["masks"].detach(),
                                                       tc.get("verifier_regions", 4))
            vl = self.model.verifier(X[:B], R, idx).float()
            kv = known.float()
            l_v = (F.binary_cross_entropy_with_logits(vl, t, reduction="none") * kv).sum() / kv.sum().clamp_min(1)
            total = total + lc["verify"] * l_v
            logs["loss_verify"] = float(l_v.detach())
            logs["verify_acc"] = float((((vl > 0).float() == t).float() * kv).sum() / kv.sum().clamp_min(1))
        return total, logs, o

    # ------------------------------------------------------------------ epochs
    def train_epoch(self, epoch: int) -> dict:
        self.model.train()
        tc = self.cfg["train"]
        bf16 = tc.get("bf16", True)
        ys, ps, cs, cps = [], [], [], []
        dacc = torch.zeros(self.K + 1, 2, device=self.device)
        sums: dict[str, float] = {}
        n = 0
        t0 = time.time()
        for it, b in enumerate(self.dl_tr):
            b = {k: (v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in b.items()}
            x = b["image"]
            B = x.shape[0]
            X = x
            if self.cf_on and b["cf_has"].any():
                mp = int(self.cfg["cf"].get("max_pairs", 16))
                if int(b["cf_has"].sum()) > mp:           # bound memory: at most max_pairs CF pairs per batch
                    keep = torch.zeros_like(b["cf_has"])
                    keep[torch.nonzero(b["cf_has"])[:mp, 0]] = True
                    b["cf_has"] = keep
                has = b["cf_has"]
                X = torch.cat([x, b["cf_minus"][has], b["cf_ctrl"][has]])
            co = None
            p_ri = tc.get("cem_randint", 0.0)
            if p_ri > 0 and self.has_concepts:
                c = b["concepts"]
                sel = (torch.rand_like(c) < p_ri) & b["concept_known"]
                co = torch.full((X.shape[0], self.K), float("nan"), device=self.device)
                co[:B] = torch.where(sel, c, torch.full_like(c, float("nan")))
            if epoch == 1 and it == 0 and getattr(self.model, "crop_mode", False):
                g = self.model.check_mask_only_path(x[:2])
                self.log(f"mask-only path check (crop evidence): max |dlogits/dF| with masks detached = {g}")
                assert g == 0.0, f"global features reach the diagnosis head outside the masks (grad {g})"
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                out = self.model(X, concept_override=co)
                loss, logs, o = self.compute_loss(out, b, B, X)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            if tc.get("grad_clip", 0) > 0:
                gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc["grad_clip"])
                logs["grad_norm"] = float(gn)
            self.opt.step()
            self.sched.step()
            logs["loss"] = float(loss.detach())
            if "n_crops" in out.get("extra", {}):
                logs["n_crops"] = float(out["extra"]["n_crops"])
            for k, v in logs.items():
                sums[k] = sums.get(k, 0.0) + v * B
            n += B
            ys.append(b["label"].cpu().numpy())
            ps.append(F.softmax(o["logits"].float().detach(), -1).cpu().numpy())
            cs.append(b["concepts"].cpu().numpy())
            cps.append(torch.sigmoid(o["concept_logits"].float().detach()).cpu().numpy())
            if self.use_mask:
                dacc += dice_score(o["masks"].detach().float(), b["masks"], b["mask_valid"])
            if it % tc.get("log_every", 20) == 0:
                self.log(f"ep {epoch} it {it}/{self.steps_per_epoch} loss {logs['loss']:.4f} "
                         + (f"crops={int(logs['n_crops'])} " if "n_crops" in logs else "")
                         + " ".join(f"{k}={v:.3f}" for k, v in logs.items() if k.startswith("loss_"))
                         + f" lr={self.sched.get_last_lr()[0]:.2e} {(time.time()-t0)/(it+1):.2f}s/it")
        m = {k: v / max(n, 1) for k, v in sums.items()}
        m.update(task_metrics(np.concatenate(ys), np.concatenate(ps)))
        m["concept_auc"] = concept_auc(np.concatenate(cs), np.concatenate(cps))
        if self.use_mask:
            d = (dacc[:, 0] / dacc[:, 1].clamp_min(1)).cpu().numpy()
            m["mask_dice"] = float(np.mean(d[:self.K]))
            m["lesion_dice"] = float(d[self.K])
        m["epoch_time"] = time.time() - t0
        m["gpu_peak_gb"] = torch.cuda.max_memory_allocated() / 1e9
        return m

    @torch.no_grad()
    def val_loss(self, d: dict) -> float:
        y = torch.tensor(d["y_true"])
        lg = torch.tensor(d["logits"])
        return float(F.cross_entropy(lg, y, weight=self.class_w.cpu() if self.class_w is not None else None))

    def evaluate(self, loader) -> tuple[dict, dict]:
        d = predict(self.model, loader, self.device, self.cfg["train"].get("bf16", True))
        m = dump_metrics(d)
        m["loss"] = self.val_loss(d)
        return m, d

    # ------------------------------------------------------------------ main loop
    def fit(self) -> dict:
        cfg = self.cfg
        name = cfg.get("run_name") or self.run_id
        summary = {"config": cfg.get("_config_file", ""), "model": {k: cfg["model"].get(k) for k in
                   ["concept_head", "diag_head", "mask_head", "detach_masks"]},
                   "verifier": (cfg["model"].get("verifier") or {}).get("mode"), "cf": self.cf_on,
                   "epochs": self.epochs, "subset": cfg["data"].get("subset", 0), "seed": cfg.get("seed", 0)}
        T.register_run(self.run_id, name, cfg.get("idea", "baseline"), self.gpu, self.epochs, config=summary)
        best, best_key, bad = {}, (-1.0, -1.0), 0
        best_epoch = 0
        try:
            last_ep = 0
            for ep in range(1, self.epochs + 1):
                last_ep = ep
                tm = self.train_epoch(ep)
                pe = getattr(self.model, "proto_project_every", 0)
                if hasattr(self.model, "prototypes") and pe > 0 and ep % pe == 0:
                    tm.update(self.project_prototypes(ep))
                T.log_metrics(self.run_id, "train", ep, **{k: v for k, v in tm.items() if np.isfinite(v)})
                vm, _ = self.evaluate(self.dl_va)
                T.log_metrics(self.run_id, "val", ep, **{k: v for k, v in vm.items() if np.isfinite(v)})
                key = (vm["bacc"], vm.get("auc", 0.0))
                self.log(f"epoch {ep}: train loss {tm['loss']:.4f} bacc {tm['bacc']:.3f} | val loss {vm['loss']:.4f} "
                         f"bacc {vm['bacc']:.3f} auc {vm.get('auc', float('nan')):.3f} "
                         f"cauc {vm.get('concept_auc', float('nan')):.3f} dice {vm.get('mask_dice', float('nan')):.3f}"
                         f" ({tm['epoch_time']:.0f}s)")
                if key > best_key:
                    best_key, bad, best_epoch = key, 0, ep
                    best = {"epoch": ep, **{f"val_{k}": round(float(v), 4) for k, v in vm.items()
                                           if k in ("bacc", "acc", "auc", "concept_auc", "mask_dice")}}
                    self.save("best.pt", ep, vm)
                    T.update_run(self.run_id, best=best)
                else:
                    bad += 1
                    if bad >= cfg["train"].get("patience", 10):
                        self.log(f"early stopping at epoch {ep} (best {best_epoch})")
                        break
            self.load("best.pt")
            if hasattr(self.model, "prototypes") and cfg["model"].get("lesion_embed", {}).get("proto_final_push", True):
                vm0, _ = self.evaluate(self.dl_va)
                pinfo = self.project_prototypes(best_epoch, tag="final")
                self.refit_last_layer()
                vm, _ = self.evaluate(self.dl_va)
                best["val_bacc_before_push"] = round(vm0["bacc"], 4)
                best["val_bacc_after_push"] = round(vm["bacc"], 4)
                self.log(f"final prototype push + last-layer refit: val bacc {vm0['bacc']:.3f} -> {vm['bacc']:.3f} "
                         f"(mean proto sim {pinfo.get('proto_mean_sim', float('nan')):.3f})")
                self.save("best.pt", best_epoch, vm)
            if self.is_report and cfg["model"].get("report", {}).get("refit", False):
                self.refit_report_head()
                vm, _ = self.evaluate(self.dl_va)
                best["val_bacc_refit"] = round(vm["bacc"], 4)
                self.save("best.pt", best_epoch, vm)
            res = self.final_dumps(best_epoch)
            best.update({f"test_{k}": round(float(v), 4) for k, v in res.items()
                         if k in ("bacc", "auc", "concept_auc", "mask_dice")})
            T.update_run(self.run_id, status="done", best=best, epoch=last_ep)   # progress = epochs trained
            self.log(f"done. best={best}")
        except Exception as e:
            self.log("FAILED:\n" + traceback.format_exc())
            T.update_run(self.run_id, status="failed", error=str(e)[:300])
            raise
        return best

    def save(self, name: str, epoch: int, vm: dict) -> None:
        torch.save({"model": self.model.state_dict(), "config": self.cfg, "epoch": epoch, "val": vm,
                    "class_names": self.class_names, "concept_names": self.concept_names}, self.dir / name)

    def load(self, name: str) -> None:
        ck = torch.load(self.dir / name, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ck["model"])

    def final_dumps(self, epoch: int) -> dict:
        extra = {}
        if hasattr(self.model, "report_names") and (self.is_report or getattr(self.model, "report_feats", False)):
            extra["report_names"] = self.model.report_names
        if hasattr(self.model, "head_input_names"):
            extra["head_input_names"] = self.model.head_input_names
        if hasattr(self.model, "proto_src"):
            extra["proto_src_ids"] = [f"ISIC_{int(v):07d}" if int(v) >= 0 else "" for v in self.model.proto_src.tolist()]
            extra["proto_src_patch"] = self.model.proto_src_patch.tolist()
        vm, dv = self.evaluate(self.dl_va)
        save_dump(self.dir / "preds_val.npz", dv, self.class_names, self.concept_names, extra)
        tm, dt = self.evaluate(self.dl_te)
        save_dump(self.dir / "preds_test.npz", dt, self.class_names, self.concept_names, extra)
        T.log_metrics(self.run_id, "test", epoch, **{k: v for k, v in tm.items() if np.isfinite(v)})
        self.log(f"TEST: {json.dumps({k: round(v, 4) for k, v in tm.items()})}")
        if self.cfg.get("eval", {}).get("ph2", True):
            try:
                self.dump_ph2()
            except Exception as e:  # PH2 is optional
                self.log(f"PH2 dump skipped: {e}")
        return tm

    @torch.no_grad()
    def dump_ph2(self) -> None:
        from cbm.data.datasets import DATASETS
        if not Path(DATASETS["ph2"]["manifest"]).exists():
            self.log("PH2 manifest missing; skipping preds_ph2.npz")
            return
        ds = PH2Dataset(self.cfg["data"]["image_size"], self.cfg["data"]["mask_size"])
        dl = torch.utils.data.DataLoader(ds, batch_size=32, num_workers=4, collate_fn=collate)
        self.model.eval()
        acc = {k: [] for k in ["image_id", "y_true", "y_prob", "c_prob", "ph2_concepts", "mask_pred", "lesion_true"]}
        for b in dl:
            x = b["image"].to(self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.model(x)
            acc["image_id"] += list(b["image_id"])
            acc["y_true"].append(b["label"].numpy())
            acc["y_prob"].append(F.softmax(out["logits"].float(), -1).cpu().numpy())
            acc["c_prob"].append(torch.sigmoid(out["concept_logits"].float()).cpu().numpy())
            acc["ph2_concepts"].append(b["ph2_concepts"].numpy())
            if bool(b["lesion_valid"].all()):
                acc["lesion_true"].append((F.adaptive_avg_pool2d(b["lesion"][:, None].float(), (64, 64))[:, 0] >= 0.5)
                                          .numpy())
            if out.get("masks") is not None:
                m64 = F.adaptive_avg_pool2d(out["masks"].float(), (64, 64))
                acc["mask_pred"].append((m64 * 255 + 0.5).clamp(0, 255).byte().cpu().numpy())
        d = {k: (np.array(v) if k == "image_id" else np.concatenate(v)) for k, v in acc.items() if v}
        if "lesion_true" in d and len(d["lesion_true"]) != len(d["y_true"]):
            d.pop("lesion_true")
        d["ph2_class_names"] = np.array(ds.info["class_names"])
        d["ph2_concept_names"] = np.array(ds.info["concept_names"])
        d["class_names"] = np.array(self.class_names)
        d["concept_names"] = np.array(self.concept_names)
        np.savez_compressed(self.dir / "preds_ph2.npz", **d)
        self.log(f"wrote preds_ph2.npz ({len(ds)} images)")

    # ------------------------------------------------------------------ prototypes (lesion channel)
    @torch.no_grad()
    def project_prototypes(self, epoch: int, tag: str = "") -> dict:
        """PIP-Net/ProtoPNet-style push: replace every prototype by its nearest training lesion embedding (pooled
        [CLS, mean] embedding, or the nearest lesion patch token for max_patch), computed from the model's own
        predicted lesion masks on the train split (eval transform). Records the source image id (numeric part) and
        patch in model buffers and writes prototypes[_<tag>].json with the 5 nearest training lesions per prototype."""
        m = self.model
        was = m.training
        m.eval()
        ds = DermDataset(self.cfg["data"]["dataset"], "train", subset=self.cfg["data"].get("subset", 0),
                         **dataset_kwargs(self.cfg))
        dl = make_loader(ds, self.cfg["train"]["batch_size"], False, self.cfg["data"].get("num_workers", 8))
        P = m.n_protos
        pr = F.normalize(m.prototypes.float(), dim=1)
        M_CAND = P + 8                     # candidates kept per prototype (distinct images) for the unique assignment
        cand: list[list[tuple]] = [[] for _ in range(P)]     # (sim, image_id, ds_idx, patch, vec_cpu)
        K = m.K
        for b in dl:
            x = b["image"].to(self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.cfg["train"].get("bf16", True)):
                f = m.features(x)
                masks = torch.sigmoid(m.mask_head(f).float())
                sel, feat, tokens, pvalid = m._lesion_crop_feats(x, masks[:, K])
            if sel.numel() == 0:
                continue
            if m.proto_pool == "pooled":
                z = m.proto_ln(feat.float())
                sim = F.normalize(z, dim=-1) @ pr.T                                  # (n, P)
                patch = torch.full_like(sim, -1, dtype=torch.long)
            else:
                z = m.proto_ln(tokens.float())                                       # (n, hw, d)
                st = F.normalize(z, dim=-1) @ pr.T                                   # (n, hw, P)
                st = torch.where(pvalid[..., None], st, torch.full_like(st, -9.0))
                sim, patch = st.max(1)                                               # (n, P)
            simc, patchc = sim.cpu(), patch.cpu()
            for j in range(P):
                for ii in range(len(sel)):
                    v = float(simc[ii, j])
                    if v <= -9.0 or (len(cand[j]) >= M_CAND and v <= cand[j][-1][0]):
                        continue
                    n_img = int(sel[ii])
                    vec = (z[ii] if m.proto_pool == "pooled" else z[ii, int(patchc[ii, j])]).detach().cpu()
                    cand[j].append((v, b["image_id"][n_img], int(b["idx"][n_img]), int(patchc[ii, j]), vec))
                    cand[j] = sorted(cand[j], key=lambda t: -t[0])[:M_CAND]
        unique = bool(self.cfg["model"].get("lesion_embed", {}).get("proto_unique_push", True))
        taken, chosen = set(), [None] * P
        for j in sorted(range(P), key=lambda j: -(cand[j][0][0] if cand[j] else -9.0)):
            for c in cand[j]:
                if not unique or c[2] not in taken:
                    chosen[j] = c
                    taken.add(c[2])
                    break
            if chosen[j] is None and cand[j]:
                chosen[j] = cand[j][0]                     # all candidates taken: allow a duplicate source
        best_sim = torch.tensor([c[0] if c else float("nan") for c in chosen])
        for j, c in enumerate(chosen):
            if c is not None:
                m.prototypes.data[j] = c[4].to(m.prototypes.device, m.prototypes.dtype)
        ids = [c[1] if c else "" for c in chosen]
        best_patch = torch.tensor([c[3] if c else -1 for c in chosen], dtype=torch.long)
        num = [int(re.sub(r"\D", "", s_) or -1) if s_ else -1 for s_ in ids]
        m.proto_src.copy_(torch.tensor(num, dtype=torch.long))
        m.proto_src_patch.copy_(best_patch)
        ok = ~torch.isnan(best_sim)
        top = [[(c[0], c[1]) for c in cand[j][:5]] for j in range(P)]
        info = {"epoch": epoch, "tag": tag, "proto_pool": m.proto_pool, "prototypes": [
            {"j": j, "src_image_id": ids[j], "src_patch": int(best_patch[j]), "sim_before_push": float(best_sim[j]),
             "nearest_train": [{"image_id": iid, "sim": s_} for s_, iid in top[j]]} for j in range(P)]}
        n_unique = len({i for i in ids if i})
        info["n_unique_sources"] = n_unique
        with open(self.dir / f"prototypes{('_' + tag) if tag else ''}.json", "w") as fo:
            json.dump(info, fo, indent=1)
        m.train(was)
        self.log(f"prototype push (epoch {epoch}{', ' + tag if tag else ''}): mean sim before push "
                 f"{float(best_sim[ok].mean()) if ok.any() else float('nan'):.3f}, {n_unique}/{P} distinct source lesions")
        return {"proto_mean_sim": float(best_sim[ok].mean()) if ok.any() else float("nan"),
                "proto_unique_sources": float(n_unique)}

    def refit_last_layer(self, steps: int = 300) -> None:
        """After the final push: refit only the linear diagnosis head on train-split head inputs (eval transform),
        class-weighted CE, same weight decay - so the head matches the projected prototypes (ProtoPNet last layer)."""
        head = self.model.diag
        if not isinstance(head, torch.nn.Linear):
            return
        ds = DermDataset(self.cfg["data"]["dataset"], "train", subset=self.cfg["data"].get("subset", 0),
                         **dataset_kwargs(self.cfg))
        dl = make_loader(ds, self.cfg["train"]["batch_size"], False, self.cfg["data"].get("num_workers", 8))
        d = predict(self.model, dl, self.device, self.cfg["train"].get("bf16", True))
        X = torch.tensor(d["head_inputs"], device=self.device)
        y = torch.tensor(d["y_true"], device=self.device)
        opt = torch.optim.AdamW(head.parameters(), lr=self.cfg["train"].get("lr_c2y", 1e-2) or 1e-2,
                                weight_decay=self.cfg["train"]["weight_decay"])
        with torch.enable_grad():
            for _ in range(steps):
                opt.zero_grad()
                loss = F.cross_entropy(head(X), y, weight=self.class_w)
                loss.backward()
                opt.step()
        self.log(f"last-layer refit: train CE {loss.item():.4f}")

    # ------------------------------------------------------------------ I3 refit
    @torch.no_grad()
    def collect_reports(self, loader):
        self.model.eval()
        R, Y = [], []
        for b in loader:
            x = b["image"].to(self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.model(x)
            R.append(out["extra"]["report"].float().cpu().numpy())
            Y.append(b["label"].numpy())
        return np.concatenate(R), np.concatenate(Y)

    def refit_report_head(self) -> None:
        """Fit an L1 logistic regression on train-set report descriptors (eval transform), choose C on val BACC,
        and load it into the linear report head (standardisation = train mean/std in the BatchNorm buffers)."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score
        head = self.model.diag
        if head.kind != "linear":
            return
        ds = DermDataset(self.cfg["data"]["dataset"], "train", subset=self.cfg["data"].get("subset", 0),
                         **dataset_kwargs(self.cfg))
        dl = make_loader(ds, self.cfg["train"]["batch_size"], False, self.cfg["data"].get("num_workers", 8))
        Rtr, Ytr = self.collect_reports(dl)
        Rva, Yva = self.collect_reports(self.dl_va)
        mu, sd = Rtr.mean(0), Rtr.std(0) + 1e-5
        Ztr, Zva = (Rtr - mu) / sd, (Rva - mu) / sd
        best = None
        for Cr in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
            lr = LogisticRegression(penalty="l1", C=Cr, solver="saga", max_iter=5000, class_weight="balanced")
            lr.fit(Ztr, Ytr)
            ba = balanced_accuracy_score(Yva, lr.predict(Zva))
            nz = int((np.abs(lr.coef_) > 1e-6).sum())
            self.log(f"report refit C={Cr}: val bacc {ba:.3f} nonzero={nz}")
            if best is None or ba > best[0] + 1e-9:
                best = (ba, Cr, lr)
        lr = best[2]
        coef = lr.coef_ if lr.coef_.shape[0] == self.C else np.vstack([-lr.coef_, lr.coef_])
        dev = head.net.weight.device
        with torch.no_grad():
            head.bn.running_mean.copy_(torch.tensor(mu, dtype=torch.float32, device=dev))
            head.bn.running_var.copy_(torch.tensor(sd ** 2 - head.bn.eps, dtype=torch.float32, device=dev).clamp_min(0))
            head.net.weight.copy_(torch.tensor(coef, dtype=torch.float32, device=dev))
            head.net.bias.copy_(torch.tensor(lr.intercept_, dtype=torch.float32, device=dev))
        self.log(f"report head refit: C={best[1]} val bacc {best[0]:.3f}")


def run(cfg: dict, run_id: str, gpu: int = 0) -> dict:
    tr = Trainer(cfg, run_id, gpu)
    best = tr.fit()
    if cfg.get("eval", {}).get("after_train", True):
        del tr.opt
        try:
            from cbm.eval.evaluate_run import evaluate_run
            evaluate_run(run_id, model=tr.model, full=cfg.get("eval", {}).get("full", True))
        except Exception:
            tr.log("evaluate_run failed:\n" + traceback.format_exc())
            T.log_event(f"evaluate_run failed for {run_id}", "error", source=f"run:{run_id}")
    return best
