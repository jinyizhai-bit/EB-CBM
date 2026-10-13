"""Pure-numpy metrics for the evaluator (DESIGN Section 7). All functions take dump arrays."""
from __future__ import annotations

import numpy as np


# ============================================================================ task
def ece_score(y: np.ndarray, prob: np.ndarray, n_bins: int = 15) -> float:
    conf = prob.max(1)
    pred = prob.argmax(1)
    corr = (pred == y).astype(np.float64)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return float(e)


def task_metrics(y: np.ndarray, prob: np.ndarray, class_names=None, mel_index: int = 1) -> dict:
    from sklearn.metrics import (balanced_accuracy_score, f1_score, log_loss, roc_auc_score)
    pred = prob.argmax(1)
    C = prob.shape[1]
    out = {"acc": float((pred == y).mean()), "bacc": float(balanced_accuracy_score(y, pred)),
           "macro_f1": float(f1_score(y, pred, average="macro", labels=list(range(C)), zero_division=0)),
           "ece": ece_score(y, prob)}
    try:
        out["nll"] = float(log_loss(y, np.clip(prob, 1e-7, 1), labels=list(range(C))))
    except ValueError:
        pass
    per = {}
    for k in range(C):
        if 0 < (y == k).sum() < len(y):
            per[k] = float(roc_auc_score(y == k, prob[:, k]))
    if per:
        out["auroc_macro"] = float(np.mean(list(per.values())))
    if mel_index in per:
        out["auroc_mel"] = per[mel_index]
    names = list(class_names) if class_names is not None else [str(k) for k in range(C)]
    out["auroc_per_class"] = {names[k]: v for k, v in per.items()}
    out["recall_per_class"] = {names[k]: float((pred[y == k] == k).mean()) for k in range(C) if (y == k).any()}
    return out


# ============================================================================ concepts
def concept_metrics(c_true: np.ndarray, c_prob: np.ndarray, names) -> dict:
    from sklearn.metrics import f1_score, roc_auc_score
    out = {"auroc": {}, "f1": {}, "acc": {}}
    if np.isnan(c_prob).all():
        return {"available": False}
    for k, n in enumerate(names):
        ok = ~np.isnan(c_true[:, k]) & ~np.isnan(c_prob[:, k])
        t, p = c_true[ok, k], c_prob[ok, k]
        if len(np.unique(t)) == 2:
            out["auroc"][n] = float(roc_auc_score(t, p))
        out["f1"][n] = float(f1_score(t, p >= 0.5, zero_division=0))
        out["acc"][n] = float(((p >= 0.5) == t).mean())
    out["mean_auroc"] = float(np.mean(list(out["auroc"].values()))) if out["auroc"] else float("nan")
    out["mean_f1"] = float(np.mean(list(out["f1"].values())))
    out["available"] = True
    return out


# ============================================================================ grounding
def grounding_metrics(mask_pred: np.ndarray, mask_true: np.ndarray, plane_names, thr: int = 128,
                      point_tol: int = 1) -> dict:
    """mask_pred (N,P,h,w) uint8 0..255, mask_true (N,P,h,w) bool."""
    from scipy.ndimage import binary_dilation
    p = mask_pred >= thr
    t = mask_true.astype(bool)
    out = {}
    for k, n in enumerate(plane_names):
        pk, tk = p[:, k], t[:, k]
        has = tk.reshape(len(tk), -1).any(1)
        inter = (pk & tk).reshape(len(pk), -1).sum(1)
        ps, ts = pk.reshape(len(pk), -1).sum(1), tk.reshape(len(tk), -1).sum(1)
        dice_img = (2 * inter / np.maximum(ps + ts, 1))[has]
        iou_img = (inter / np.maximum(ps + ts - inter, 1))[has]
        gi, gd = inter.sum(), ps.sum() + ts.sum()
        # pointing game: argmax of the soft predicted map inside the (dilated) expert mask
        hits = []
        for i in np.flatnonzero(has):
            j = np.unravel_index(np.argmax(mask_pred[i, k]), mask_pred.shape[-2:])
            tt = binary_dilation(tk[i], iterations=point_tol) if point_tol > 0 else tk[i]
            hits.append(bool(tt[j]))
        out[n] = {"dice": float(dice_img.mean()) if has.any() else float("nan"),
                  "iou": float(iou_img.mean()) if has.any() else float("nan"),
                  "dice_global": float(2 * gi / gd) if gd > 0 else float("nan"),
                  "iou_global": float(gi / max(gd - gi, 1)) if gd > 0 else float("nan"),
                  "pointing": float(np.mean(hits)) if hits else float("nan"),
                  "fp_rate_absent": float((ps[~has] > 0).mean()) if (~has).any() else float("nan"),
                  "n_present": int(has.sum())}
    K = len(plane_names) - 1
    for m in ("dice", "iou", "dice_global", "iou_global", "pointing"):
        out[f"mean_{m}"] = float(np.nanmean([out[n][m] for n in plane_names[:K]]))
    return out


# ============================================================================ leakage
def label_from_concepts_cv(c: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 0) -> float:
    """Balanced accuracy of a 5-fold CV logistic regression predicting y from concept features c (N,D)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.model_selection import StratifiedKFold
    c = np.nan_to_num(c.reshape(len(c), -1).astype(np.float64), nan=0.0)
    pred = np.zeros_like(y)
    nf = min(folds, np.bincount(y).min()) if np.bincount(y).min() >= 2 else 2
    for tr, te in StratifiedKFold(nf, shuffle=True, random_state=seed).split(c, y):
        lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(c[tr], y[tr])
        pred[te] = lr.predict(c[te])
    return float(balanced_accuracy_score(y, pred))


def leakage_cv(c_true: np.ndarray, c_prob: np.ndarray, y: np.ndarray, c_repr: np.ndarray | None = None) -> dict:
    """Mahinpei/Havasi-style leakage probe: how much more label information do the *soft* concept scores (or
    the full per-concept representation fed to the head) carry than their hard (thresholded) versions?"""
    ok = ~np.isnan(c_true).any(1)
    out = {"bacc_from_gt": label_from_concepts_cv(c_true[ok], y[ok]),
           "bacc_from_hard": label_from_concepts_cv((c_prob[ok] >= 0.5).astype(float), y[ok]),
           "bacc_from_soft": label_from_concepts_cv(c_prob[ok], y[ok])}
    out["soft_minus_hard"] = out["bacc_from_soft"] - out["bacc_from_hard"]
    if c_repr is not None and c_repr.shape[-1] > 1:
        out["bacc_from_repr"] = label_from_concepts_cv(c_repr[ok], y[ok])
        out["repr_minus_hard"] = out["bacc_from_repr"] - out["bacc_from_hard"]
    return out


def ois(c_true: np.ndarray, c_repr: np.ndarray, seed: int = 0, hidden: int = 32) -> dict:
    """Oracle Impurity Score (Espinosa Zarlenga et al., AAAI 2023, "Towards Robust Metrics for Concept
    Representation Evaluation").  pi(R, C)_ij = AUROC of a helper MLP trained to predict concept j from the
    representation of concept i (2-fold cross-fitting on the given split); oracle pi(C, C) uses the ground-truth
    concept i as input.  OIS = 2/k * ||pi(C_hat, C) - pi(C, C)||_F  in [0, 1] (lower = purer).
    c_true (N,k) binary (rows with NaN dropped), c_repr (N,k,d)."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.neural_network import MLPClassifier
    ok = ~np.isnan(c_true).any(1) & ~np.isnan(c_repr.reshape(len(c_repr), -1)).any(1)
    C, R = c_true[ok].astype(int), c_repr[ok].astype(np.float64)
    if R.ndim == 2:
        R = R[..., None]
    N, k = C.shape

    def purity(inp_fn):
        P = np.full((k, k), 0.5)
        for j in range(k):
            yj = C[:, j]
            if len(np.unique(yj)) < 2 or np.bincount(yj).min() < 4:
                continue
            for i in range(k):
                X = inp_fn(i)
                X = (X - X.mean(0)) / (X.std(0) + 1e-6)
                score = np.zeros(N)
                for tr, te in StratifiedKFold(2, shuffle=True, random_state=seed).split(X, yj):
                    clf = MLPClassifier((hidden,), max_iter=400, random_state=seed, early_stopping=False)
                    clf.fit(X[tr], yj[tr])
                    score[te] = clf.predict_proba(X[te])[:, 1]
                a = roc_auc_score(yj, score)
                P[i, j] = max(a, 1 - a)
        return P

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pr = purity(lambda i: R[:, i, :])
        po = purity(lambda i: C[:, i:i + 1].astype(np.float64))
    return {"ois": float(2.0 / k * np.linalg.norm(pr - po)), "purity_pred": pr.round(4).tolist(),
            "purity_oracle": po.round(4).tolist()}


# ============================================================================ selective prediction
def accuracy_coverage(correct: np.ndarray, score: np.ndarray) -> dict:
    """Sort by confidence score (high first); accuracy of the retained set as coverage grows."""
    o = np.argsort(-score, kind="stable")
    c = correct[o].astype(np.float64)
    acc = np.cumsum(c) / np.arange(1, len(c) + 1)
    cov = np.arange(1, len(c) + 1) / len(c)
    i80 = int(np.ceil(0.8 * len(c))) - 1
    i50 = int(np.ceil(0.5 * len(c))) - 1
    trap = float(np.sum((acc[1:] + acc[:-1]) * np.diff(cov)) / 2) if len(c) > 1 else 0.0   # numpy-version agnostic
    return {"auc": trap / (cov[-1] - cov[0]) if len(c) > 1 else float(acc[-1]),
            "acc@80": float(acc[i80]), "acc@50": float(acc[i50]),
            "curve_cov": cov[::max(1, len(c) // 50)].round(4).tolist(),
            "curve_acc": acc[::max(1, len(c) // 50)].round(4).tolist()}


def balanced_accuracy(y: np.ndarray, pred: np.ndarray, C: int | None = None) -> float:
    C = C or int(max(y.max(), pred.max()) + 1)
    r = [(pred[y == k] == k).mean() for k in range(C) if (y == k).any()]
    return float(np.mean(r))


def macro_auroc(y: np.ndarray, prob: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    a = [roc_auc_score(y == k, prob[:, k]) for k in range(prob.shape[1]) if 0 < (y == k).sum() < len(y)]
    return float(np.mean(a)) if a else float("nan")


def mean_concept_auroc(c_true: np.ndarray, c_prob: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    a = []
    for k in range(c_true.shape[1]):
        ok = ~np.isnan(c_true[:, k])
        if len(np.unique(c_true[ok, k])) == 2:
            a.append(roc_auc_score(c_true[ok, k], c_prob[ok, k]))
    return float(np.mean(a)) if a else float("nan")
