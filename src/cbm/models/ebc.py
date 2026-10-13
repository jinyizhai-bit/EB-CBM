"""Unified evidence-bound CBM (EBCModel): every Stage-A method is a combination of config flags.

model config keys (see configs/base.yaml):
  backbone        dict for build_backbone
  mask_head       bool     K+1 soft masks (128x128) supervised with BCE + soft Dice
  concept_head    none | global (Koh) | cem | rr (I2 region-restricted)
  diag_head       blackbox | probs (Koh joint) | hard (I2) | soft_rr (I2) | global (CEM mixing) | report (I3)
  detach_masks    stop-gradient from the concept/diagnosis path into the masks (I2/I3)
  verifier        {enabled, mode: none|weight|input}  (I4)
  report          {head: linear|mlp, l1, colour_detail, refit}   (I3)
  report_feats    bool: concatenate the I3 report descriptors (computed on *detached* masks, presence entries
                  dropped, standardised by a non-affine BatchNorm) into the diagnosis-head input of any head
  backbone_input_size  optional int: the model input x (e.g. 512 = full cache resolution) is resized to this size
                  for the global backbone only; crops (concept evidence, verifier) are taken from x itself
  concept_evidence {source: tokens|crop, ...}: tokens = RRConceptHead pooling of backbone features inside M_c;
                  crop = CropEvidence: shared local encoder (timm resnet18) on masked square crops of x for every
                  concept (+ lesion) mask. With crop, the global backbone reaches the diagnosis head ONLY through the
                  masks (asserted structurally at init and by a gradient test, ``check_mask_only_path``)
  discover        {n_parts, app_dim, support_tau, bias_init, lesion_gate, shared_decoder, detach_features, reg}:
                  crop mode only. A light part decoder (or, shared_decoder, the mask head) emits n_parts extra soft
                  masks (planes K+1..K+n_parts, no expert labels), multiplied by the hard detached predicted lesion
                  mask (lesion_gate); each is read like a concept but the crop pixels are
                  weighted by the upsampled SOFT mask (exactly 0 where <= support_tau), so the task gradient reaches
                  the part masks; local encoder -> app_dim appearance + log-area per part (no presence logit).
                  Regularisers in heads.part_regularizers (weights in discover.reg, applied by the trainer)
  lesion_embed    {enabled, dim, source: tokens|crop|crop_backbone|crop_backbone_proto, n_protos, proto_pool,
                  proto_temp, proto_div, proto_project_every}: crop_backbone_proto = cosine similarities between the
                  LayerNorm'd lesion embedding of the second pass (pooled, or max over lesion patches) and P learned
                  prototypes, projected onto training lesion embeddings during training (trainer); crop_backbone (crop
                  mode) = masked lesion crop
                  through the same global backbone in a second pass ([CLS, mean tokens] -> LN-Linear-tanh to dim);
  lesion_embed    {enabled, dim, source: tokens|crop}: per-image appearance descriptor restricted to the predicted
                  lesion mask, appended to the diagnosis-head input; follows mask_override of the lesion plane (K).
                  tokens: masked avg + max of a 1x1 projection of F (ViT tokens still carry global context);
                  crop: small CNN on the lesion crop with all non-lesion pixels zeroed (strictly pixel-local)

Contract (DESIGN Section 9): ``out = model(x, concept_override=None, mask_override=None)`` ->
  {"logits" (N,C), "concept_logits" (N,K) [NaN if no concepts], "masks" (N,K+1,128,128) probabilities
   [None if no mask head], "extra": {...}}.
``concept_override`` (N,K) NaN = keep; replaces the concept probability fed to the diagnosis head (and, for
region heads, zeroes area/appearance/report descriptors of concepts set to 0; uses the train-prior area for
concepts set to 1 whose predicted mask is empty).  ``mask_override`` (N,K+1,h,w) NaN planes = keep; replaces
the predicted mask planes (any resolution, resized to 128) *before* pooling, so concept scores, areas,
descriptors, verification and the diagnosis are recomputed from the overriding masks (spatial intervention).
``forward_heads(F, x, ...)`` runs everything after the backbone (evaluator caches F).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import build_backbone
from .heads import (CEMHead, CropEvidence, square_boxes, GlobalConceptHead, LesionCropEmbed, LesionEmbed, MaskHead, ReportHead, RRConceptHead, Verifier,
                    absent_struct_values, compute_report, report_names)


def _denorm(x: torch.Tensor) -> torch.Tensor:
    m = torch.tensor([0.485, 0.456, 0.406], device=x.device, dtype=torch.float32).view(1, 3, 1, 1)
    s = torch.tensor([0.229, 0.224, 0.225], device=x.device, dtype=torch.float32).view(1, 3, 1, 1)
    return (x.float() * s + m).clamp(0, 1)


class EBCModel(nn.Module):
    def __init__(self, cfg: dict, n_classes: int, concept_names: list[str]):
        super().__init__()
        self.cfg = cfg
        self.C, self.K = n_classes, len(concept_names)
        self.concept_names = list(concept_names)
        K, C = self.K, self.C
        self.backbone = build_backbone(cfg.get("backbone", {}))
        d = self.backbone.out_dim
        self.mask_size = int(cfg.get("mask_size", 128))
        self.use_mask = bool(cfg.get("mask_head", False))
        self.ch = cfg.get("concept_head", "none")
        self.dh = cfg.get("diag_head", "blackbox")
        self.detach_masks = bool(cfg.get("detach_masks", True))
        hid = int(cfg.get("diag_hidden", 0))
        if self.ch == "rr" or self.dh == "report":
            assert self.use_mask, "rr concept head / report head need mask_head: true"
        dcfg = cfg.get("discover", {}) or {}
        self.n_parts = int(dcfg.get("n_parts", 0) or 0)
        self.part_lesion_gate = bool(dcfg.get("lesion_gate", True))
        # separate light decoder for the part planes (default): the part objectives cannot disturb the expert mask
        # decoder; with detach_features the backbone is not pulled by them either
        self.part_shared = bool(dcfg.get("shared_decoder", False))
        self.part_detach_f = bool(dcfg.get("detach_features", True))
        if self.use_mask:
            # planes: K concepts, lesion (K), then n_parts discovered evidence regions (no expert labels)
            self.mask_head = MaskHead(d, K + 1 + (self.n_parts if self.part_shared else 0), out_size=self.mask_size)
            if self.n_parts > 0 and not self.part_shared:
                self.part_head = MaskHead(d, self.n_parts, mid=int(dcfg.get("decoder_dim", 128)), out_size=self.mask_size)
        rr = cfg.get("rr", {})
        self.app_dim = int(cfg.get("app_dim", 4)) if self.dh == "soft_rr" else 0
        self.bb_size = cfg.get("backbone_input_size")
        ecfg = cfg.get("concept_evidence", {}) or {}
        self.evidence = ecfg.get("source", "tokens")
        self.crop_mode = self.evidence == "crop"
        lcfg = cfg.get("lesion_embed", {}) or {}
        if self.crop_mode:
            # structural guarantee: no module may read the backbone features except the mask head
            assert self.ch == "rr", "concept_evidence=crop needs concept_head: rr"
            assert self.dh in ("probs", "hard", "soft_rr", "report"), \
                f"concept_evidence=crop: diag_head {self.dh!r} would read global features"
            assert self.use_mask, "concept_evidence=crop needs mask_head"
            if lcfg.get("enabled", False):
                assert lcfg.get("source", "tokens") in ("crop", "crop_backbone", "crop_backbone_proto"), \
                    "concept_evidence=crop needs lesion_embed.source in {crop, crop_backbone, crop_backbone_proto}"
            shared_lesion = lcfg.get("enabled", False) and lcfg.get("source", "tokens") == "crop"
            lesion_dim = int(lcfg.get("dim", 8)) if shared_lesion else 0
            self.concept_head = CropEvidence(K, self.app_dim, lesion_dim, ecfg.get("local_encoder"),
                                             int(ecfg.get("crop_size", 224)), float(ecfg.get("margin", 0.10)),
                                             float(ecfg.get("min_area_frac", 0.002)), float(ecfg.get("mask_thresh", 0.5)),
                                             bool(ecfg.get("mask_channel", True)), self.n_parts,
                                             int(dcfg.get("app_dim", 4)), float(dcfg.get("support_tau", 0.1)))
            # start with (nearly) empty masks -> null tokens, no flood of noisy full-image crops in the first steps
            nn.init.constant_(self.mask_head.out.bias, float(ecfg.get("mask_bias_init", -3.0)))
            if self.n_parts > 0:   # discovered parts start undecided (p~0.5) so that crops, and gradients, exist
                with torch.no_grad():
                    if self.part_shared:
                        self.mask_head.out.bias[K + 1:] = float(dcfg.get("bias_init", 0.0))
                    else:
                        self.part_head.out.bias.fill_(float(dcfg.get("bias_init", 0.0)))
        else:
            assert self.n_parts == 0, "discover.n_parts > 0 needs concept_evidence.source=crop"
        if self.crop_mode:
            pass
        elif self.ch == "rr":
            self.concept_head = RRConceptHead(d, K, rr.get("proj_dim", 256), rr.get("hidden", 128), self.app_dim,
                                              rr.get("pool_tau", 0.05), rr.get("null_thresh", 0.5))
        elif self.ch == "global":
            self.concept_head = GlobalConceptHead(d, K, int(cfg.get("concept_hidden", 0)))
        elif self.ch == "cem":
            self.concept_head = CEMHead(d, K, int(cfg.get("cem_dim", 16)))
        elif self.ch != "none":
            raise ValueError(self.ch)
        if self.dh == "global" and self.ch != "cem":
            self.cem_emb = CEMHead(d, K, int(cfg.get("cem_dim", 16)))   # I2-global control: CEM embeddings
        vcfg = cfg.get("verifier", {}) or {}
        self.use_verifier = bool(vcfg.get("enabled", False))
        self.verify_mode = vcfg.get("mode", "none") if self.use_verifier else "none"
        if self.use_verifier:
            assert self.use_mask, "verifier needs mask_head"
            self.verifier = Verifier(K, vcfg.get("width", 32), vcfg.get("crop", 128),
                                     strict=bool(vcfg.get("strict", self.crop_mode)))
        n_v = K if self.verify_mode == "input" else 0
        rcfg = cfg.get("report", {}) or {}
        self.colour_detail = bool(rcfg.get("colour_detail", True))
        self.report_names = report_names(self.concept_names, self.colour_detail)

        def lin(i):
            return nn.Linear(i, C) if hid == 0 else nn.Sequential(nn.Linear(i, hid), nn.GELU(), nn.Linear(hid, C))
        # optional extra evidence (defaults off -> identical modules / state dicts for the original configs)
        self.report_feats = bool(cfg.get("report_feats", False)) and self.dh not in ("report", "blackbox")
        if self.report_feats:
            assert self.use_mask, "report_feats needs mask_head: true"
            self.rep_feat_idx = [i for i, n in enumerate(self.report_names) if not n.endswith(":presence")]
            self.rep_bn = nn.BatchNorm1d(len(self.rep_feat_idx), affine=False, momentum=0.05)
        self.use_lesion = bool(lcfg.get("enabled", False)) and self.dh != "blackbox"
        if self.use_lesion:
            assert self.use_mask, "lesion_embed needs mask_head: true"
            self.lesion_dim = int(lcfg.get("dim", 8))
            self.lesion_source = lcfg.get("source", "tokens")
            if self.crop_mode and self.lesion_source == "crop":
                self.lesion_source = "shared_crop"        # lesion crop through the shared local encoder (CropEvidence)
            elif self.lesion_source in ("crop_backbone", "crop_backbone_proto"):
                # masked lesion crop encoded by the SAME global backbone in a second forward pass
                assert self.crop_mode, "lesion_embed.source=crop_backbone* is implemented for concept_evidence=crop"
                bdim = d * (2 if getattr(self.backbone, "n_prefix", 0) else 1)    # CLS + mean tokens (ViT) or mean
                if self.lesion_source == "crop_backbone":
                    self.lesion_bb_head = nn.Sequential(nn.LayerNorm(bdim), nn.Linear(bdim, self.lesion_dim),
                                                        nn.Tanh())
                    self.lesion_bb_null = nn.Parameter(torch.zeros(self.lesion_dim))
                else:
                    # interpretable lesion channel: cosine similarity to P learned prototypes (projected onto training
                    # lesion embeddings during training), pooled [CLS, mean] embedding or max over lesion patches
                    self.n_protos = int(lcfg.get("n_protos", 16))
                    self.proto_pool = lcfg.get("proto_pool", "pooled")
                    assert self.proto_pool in ("pooled", "max_patch"), self.proto_pool
                    pdim = bdim if self.proto_pool == "pooled" else d
                    self.proto_ln = nn.LayerNorm(pdim)
                    self.prototypes = nn.Parameter(F.normalize(torch.randn(self.n_protos, pdim), dim=1))
                    self.proto_temp_learned = lcfg.get("proto_temp", "learned") == "learned"
                    t0 = math.log(float(lcfg.get("proto_temp_init", 1.0)))
                    if self.proto_temp_learned:
                        self.proto_log_scale = nn.Parameter(torch.tensor(t0))
                    else:
                        self.register_buffer("proto_log_scale", torch.tensor(t0))
                    self.proto_null = nn.Parameter(torch.zeros(self.n_protos))
                    # numeric part of the source training image id (ISIC_<n>) and source patch (max_patch) per prototype
                    self.register_buffer("proto_src", torch.full((self.n_protos,), -1, dtype=torch.long))
                    self.register_buffer("proto_src_patch", torch.full((self.n_protos,), -1, dtype=torch.long))
                    self.lesion_dim = self.n_protos
                    self.proto_div = float(lcfg.get("proto_div", 0.1))
                    self.proto_project_every = int(lcfg.get("proto_project_every", 5))
                self.lesion_crop_size = int(lcfg.get("crop_size", self.bb_size or 448))
                self.lesion_margin = float(lcfg.get("margin", ecfg.get("margin", 0.10)))
                self.lesion_min_area = float(ecfg.get("min_area_frac", 0.002))
                self.lesion_thr = float(ecfg.get("mask_thresh", 0.5))
            elif self.lesion_source == "tokens":
                self.lesion_embed = LesionEmbed(d, self.lesion_dim, int(lcfg.get("proj_dim", 64)),
                                                rr.get("pool_tau", 0.05), rr.get("null_thresh", 0.5))
            elif self.lesion_source == "crop":
                self.lesion_embed = LesionCropEmbed(self.lesion_dim, int(lcfg.get("width", 32)),
                                                    int(lcfg.get("crop", 128)))
            else:
                raise ValueError(f"lesion_embed.source {self.lesion_source!r}")
        self.head_input_names = self._build_head_input_names(d, n_v)
        n_in = len(self.head_input_names)
        if self.dh == "blackbox":
            self.drop = nn.Dropout(float(cfg.get("dropout", 0.0)))
            self.diag = nn.Linear(d, C)
        elif self.dh in ("probs", "hard", "soft_rr", "global"):
            self.diag = lin(n_in)
        elif self.dh == "report":
            self.diag = ReportHead(n_in, C, rcfg.get("head", "linear"), int(rcfg.get("hidden", 32)))
        else:
            raise ValueError(self.dh)
        # mean predicted area (fraction of image) of positives, used when an intervention says "present"
        self.register_buffer("area_prior", torch.full((K,), 0.02))

    # ------------------------------------------------------------------ helpers
    def _build_head_input_names(self, d: int, n_v: int) -> list[str]:
        """'group:detail' name of every column of the diagnosis-head input (dumped as head_inputs).
        Groups: global, concept, area, appearance, cem_emb, report, verify, lesion."""
        cn, K = self.concept_names, self.K
        if self.dh == "blackbox":
            return [f"global:{i}" for i in range(d)]
        names: list[str] = []
        if self.dh in ("probs", "hard", "soft_rr"):
            names += [f"concept:{c}" for c in cn]
        if self.dh in ("hard", "soft_rr"):
            names += [f"area:{c}" for c in cn]
        if self.dh == "soft_rr":
            names += [f"appearance:{c}:{j}" for c in cn for j in range(self.app_dim)]
        if self.dh == "global":
            names += [f"cem_emb:{c}:{j}" for c in cn for j in range(int(self.cfg.get("cem_dim", 16)))]
        if self.dh == "report":
            for n in self.report_names:
                grp = "concept" if n.endswith(":presence") else ("area" if n.endswith(":area_frac_lesion")
                                                                 else "report")
                names.append(f"{grp}:{n}")
        names += [f"verify:{c}" for c in cn][:n_v]
        if self.report_feats:
            names += [f"report:{self.report_names[i]}" for i in self.rep_feat_idx]
        if self.use_lesion:
            pre = "proto" if getattr(self, "lesion_source", "") == "crop_backbone_proto" else ""
            names += [f"lesion:{pre}{j}" for j in range(self.lesion_dim)]
        if self.n_parts > 0:
            na = int((self.cfg.get("discover", {}) or {}).get("app_dim", 4))
            names += [f"part:p{j}:app{i}" for j in range(self.n_parts) for i in range(na)]
            names += [f"part:p{j}:logarea" for j in range(self.n_parts)]
        return names

    @property
    def c2y_linear(self) -> bool:
        """True if the diagnosis head is a single linear layer on concept probabilities / areas / appearance /
        report / lesion descriptors (inputs mostly in [0,1] or standardised) -> gets its own LR (train.lr_c2y)."""
        if self.dh in ("probs", "hard", "soft_rr"):
            return isinstance(self.diag, nn.Linear)
        if self.dh == "report":
            return getattr(self.diag, "kind", "") == "linear"
        return False

    def param_groups(self, lr_backbone: float, lr_heads: float, wd: float, lr_local: float | None = None,
                     lr_c2y: float | None = None) -> list[dict]:
        bb, heads, loc, c2y = [], [], [], []
        use_c2y = lr_c2y is not None and self.c2y_linear
        for n, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith("backbone."):
                bb.append(p)
            elif n.startswith("concept_head.enc.") and lr_local is not None:
                loc.append(p)
            elif n.startswith("diag.") and use_c2y:
                c2y.append(p)
            else:
                heads.append(p)
        g = [{"params": bb, "lr": lr_backbone, "weight_decay": wd, "name": "backbone"},
             {"params": heads, "lr": lr_heads, "weight_decay": wd, "name": "heads"}]
        if loc:
            g.append({"params": loc, "lr": lr_local, "weight_decay": wd, "name": "local_encoder"})
        if c2y:
            g.append({"params": c2y, "lr": lr_c2y, "weight_decay": wd, "name": "c2y"})
        return g

    @property
    def has_concepts(self) -> bool:
        return self.ch != "none"

    def _apply_mask_override(self, masks: torch.Tensor, mo: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mo = mo.to(masks.device, torch.float32)
        if mo.shape[1] < masks.shape[1]:          # e.g. expert K+1 planes for a model with discovered part planes
            pad = torch.full((mo.shape[0], masks.shape[1] - mo.shape[1]) + tuple(mo.shape[2:]), float("nan"),
                             device=mo.device)
            mo = torch.cat([mo, pad], 1)
        plane_ovr = ~torch.isnan(mo).flatten(2).any(-1)                       # (N,P) whole-plane override
        mo = torch.nan_to_num(mo, 0.0)
        if mo.shape[-1] != masks.shape[-1]:
            mo = F.interpolate(mo, size=masks.shape[-2:], mode="area" if mo.shape[-1] > masks.shape[-1]
                               else "bilinear", align_corners=None if mo.shape[-1] > masks.shape[-1] else False)
        return torch.where(plane_ovr[:, :, None, None], mo.clamp(0, 1), masks), plane_ovr

    # ------------------------------------------------------------------ forward
    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Global backbone features (x resized to backbone_input_size if set)."""
        if self.bb_size and x.shape[-1] != int(self.bb_size):
            x = F.interpolate(x, size=(int(self.bb_size), int(self.bb_size)), mode="bilinear", align_corners=False,
                              antialias=True)
        return self.backbone(x)

    def forward(self, x, concept_override=None, mask_override=None, report_override=None):
        f = self.features(x)
        return self.forward_heads(f, x, concept_override, mask_override, report_override)

    def forward_heads(self, f, x, concept_override=None, mask_override=None, report_override=None,
                      _detach_masks_all: bool = False):
        N, K = f.shape[0], self.K
        extra: dict = {}
        # globally pooled features are only computed for heads that are allowed to read them (never in crop mode)
        g = None if self.crop_mode else f.float().mean((2, 3))
        masks = None
        if self.use_mask:
            mlog = self.mask_head(f).float()
            if self.n_parts > 0 and not self.part_shared:
                plog = self.part_head(f.detach() if self.part_detach_f else f).float()
                mlog = torch.cat([mlog, plog], 1)
            extra["mask_logits"] = mlog
            masks = torch.sigmoid(mlog)
            if mask_override is not None:
                masks, povr = self._apply_mask_override(masks, mask_override)
                extra["mask_overridden"] = povr
        if masks is not None and self.n_parts > 0 and self.part_lesion_gate:
            # discovered parts live inside the predicted lesion by construction: hard, detached lesion gate (the part
            # losses can then never pull on the lesion plane, and parts cannot spread over the background)
            gate = (masks[:, K:K + 1].detach() > 0.5).float()
            masks = torch.cat([masks[:, :K + 1], masks[:, K + 1:] * gate], 1)
        Mu = None
        if masks is not None:
            if _detach_masks_all:            # gradient isolation test: cut the (allowed) mask route
                masks = masks.detach()
            Mu = masks.detach() if self.detach_masks else masks
        # ---------------- concepts
        area = app = cpos = cneg = None
        crop_lesion = None
        part_info = None
        if self.ch == "rr" and self.crop_mode:
            planes = Mu[:, :K + 1] if self.lesion_source_is("shared_crop") else Mu[:, :K]
            Mp = masks[:, K + 1:K + 1 + self.n_parts] if self.n_parts > 0 else None   # soft, gradient kept
            s, info = self.concept_head(x, planes, Mp)
            area, app = info["area"], info["app"]
            extra["rr_present"] = info["present"]
            extra["n_crops"] = torch.tensor(float(info["n_crops"] + info.get("n_part_crops", 0)), device=f.device)
            crop_lesion = info.get("lesion_emb")
            if self.n_parts > 0:
                part_info = info
                extra["part_app"] = info["part_app"]
                lesion_soft = masks[:, K].float().sum((1, 2)).clamp_min(1e-4)
                extra["part_area"] = (Mp.float().sum((2, 3)) / lesion_soft[:, None]).clamp(max=1.0)
        elif self.ch == "rr":
            s, info = self.concept_head(f, Mu[:, :K])
            area, app = info["area"], info["app"]
            extra["rr_present"] = info["present"]
        elif self.ch == "global":
            s = self.concept_head(g)
        elif self.ch == "cem":
            s, cpos, cneg = self.concept_head(g)
        else:
            s = torch.full((N, K), float("nan"), device=f.device)
        s = s.float()
        p = torch.sigmoid(s)
        co_mask = None
        if concept_override is not None and self.has_concepts:
            co = concept_override.to(p.device, torch.float32)
            co_mask = ~torch.isnan(co)
            p = torch.where(co_mask, torch.nan_to_num(co, 0.0), p)
            if area is not None:
                set0 = co_mask & (co < 0.5)
                set1 = co_mask & (co >= 0.5)
                area = torch.where(set0, torch.zeros_like(area), area)
                area = torch.where(set1 & (area < 1e-4), self.area_prior[None].expand_as(area), area)
                if app is not None and app.shape[-1] > 0:
                    app = torch.where(set0[..., None], self.concept_head.null_appearance()[None].expand_as(app), app)
        # ---------------- verifier
        v = None
        if self.use_verifier:
            v = torch.sigmoid(self.verifier.score_masks(x, masks[:, :K].detach()).float())
            extra["verify"] = v
            if co_mask is not None:
                v = torch.where(co_mask, torch.ones_like(v), v)
        p_in = p
        if self.verify_mode == "weight":
            p_in = p * v.detach()
        vin = [v.detach()] if self.verify_mode == "input" else []
        # ---------------- diagnosis (head input = concatenation of named blocks, see head_input_names)
        if self.dh == "blackbox":
            hin = g
            logits = self.diag(self.drop(g))
        else:
            blocks = []
            if self.dh == "probs":
                blocks = [p_in]
            elif self.dh == "hard":
                blocks = [p_in, area.clamp_min(0).sqrt()]
            elif self.dh == "soft_rr":
                blocks = [p_in, area.clamp_min(0).sqrt(), app.flatten(1)]
            elif self.dh == "global":
                if cpos is None:
                    _, cpos, cneg = self.cem_emb(g)
                e = p_in[..., None] * cpos + (1 - p_in[..., None]) * cneg
                blocks = [e.flatten(1)]
                extra["concept_emb"] = e
            elif self.dh == "report":
                rep = self._report(Mu[:, :K + 1], p_in, x, co_mask, report_override)
                extra["report"] = rep
                blocks = [rep]
            blocks += vin
            if self.report_feats:
                rep = self._report(masks[:, :K + 1].detach(), p_in, x, co_mask, report_override)
                extra["report"] = rep
                blocks.append(self.rep_bn(rep[:, self.rep_feat_idx]))
            if self.use_lesion:
                if self.lesion_source == "shared_crop":
                    le = crop_lesion
                elif self.lesion_source == "crop_backbone":
                    le = self._lesion_backbone_embed(x, Mu[:, K])
                elif self.lesion_source == "crop_backbone_proto":
                    le, psim = self._lesion_proto_embed(x, Mu[:, K])
                    extra["proto_sim"] = psim
                elif self.lesion_source == "tokens":
                    le = self.lesion_embed(f, Mu[:, K])
                else:
                    le = self.lesion_embed(x, Mu[:, K])
                extra["lesion_emb"] = le
                blocks.append(le)
            if part_info is not None:
                blocks += [part_info["part_app"].flatten(1), part_info["part_logarea"]]
            hin = torch.cat(blocks, -1)
            logits = self.diag(hin)
        extra["head_inputs"] = hin
        if area is not None:
            extra["area_img"] = area
        if masks is not None:
            lesion_area = masks[:, K].mean((1, 2))
            extra["lesion_area_img"] = lesion_area
            extra["area_lesion"] = (masks[:, :K].mean((2, 3)) / lesion_area[:, None].clamp_min(1e-4)).clamp(max=1.0)
        extra["concept_probs_used"] = p_in
        extra["concept_repr"] = self._concept_repr(p_in, area, app, extra.get("concept_emb"))
        return {"logits": logits.float(), "concept_logits": s, "masks": masks, "extra": extra}

    def _report(self, M, p_in, x, co_mask, report_override):
        K = self.K
        rep = compute_report(M, p_in, _denorm(x), self.colour_detail, hard_colour=self.crop_mode)
        if co_mask is not None:
            rep = absent_struct_values(rep, K, co_mask & (p_in < 0.5))
            pres_idx = torch.arange(K, device=rep.device) * 5      # presence entry must equal the override
            rep[:, pres_idx] = p_in
        if report_override is not None:
            ro = report_override.to(rep.device, torch.float32)
            rep = torch.where(torch.isnan(ro), rep, ro)
        return rep

    def lesion_source_is(self, src: str) -> bool:
        return self.use_lesion and getattr(self, "lesion_source", None) == src

    def _lesion_crop_feats(self, x: torch.Tensor, L: torch.Tensor):
        """Second backbone pass on the masked lesion crops. Returns sel (indices of images with a non-empty lesion),
        pooled [CLS, mean] features (n_sel, 2d), patch tokens (n_sel, hw, d) and patch validity (n_sel, hw) =
        >= 50 % of the patch's pixels inside the lesion region."""
        from torchvision.ops import roi_align
        H, W = x.shape[-2:]
        Lb = (L.detach().float() > self.lesion_thr).float()
        empty = Lb.flatten(1).mean(-1) < self.lesion_min_area
        sel = torch.nonzero(~empty, as_tuple=True)[0]
        if sel.numel() == 0:
            return sel, None, None, None
        S = self.lesion_crop_size
        bx = square_boxes(Lb[sel], (H, W), self.lesion_margin)
        xc = roi_align(x.float(), torch.cat([sel.float()[:, None], bx], 1), (S, S), 1.0, 0, True)
        own = torch.arange(len(sel), device=x.device, dtype=torch.float32)[:, None]
        mc = roi_align(Lb[sel][:, None], torch.cat([own, bx], 1), (S, S), Lb.shape[-1] / W, 0, True)
        keep = (mc >= 0.999).float()
        f2 = self.backbone((xc * keep).to(x.dtype))                          # second pass, shared weights
        pooled = f2.float().mean((2, 3))
        cls = getattr(self.backbone, "last_cls", None)
        feat = torch.cat([cls.float(), pooled], -1) if (cls is not None and getattr(self.backbone, "n_prefix", 0)) \
            else pooled
        h, w = f2.shape[-2:]
        tokens = f2.float().flatten(2).transpose(1, 2)                       # (n_sel, hw, d)
        pvalid = (F.adaptive_avg_pool2d(keep, (h, w)) >= 0.5).flatten(1)     # (n_sel, hw)
        return sel, feat, tokens, pvalid

    def _lesion_proto_embed(self, x: torch.Tensor, L: torch.Tensor):
        """Prototype similarities of the masked lesion crop: cos(LN(embedding), prototype_p), embedding = pooled
        [CLS, mean] or, for max_patch, the max over lesion patches. Returns (scaled similarities, raw cosines)."""
        N, P = x.shape[0], self.n_protos
        raw = self.proto_null[None].expand(N, -1).clone()
        sel, feat, tokens, pvalid = self._lesion_crop_feats(x, L)
        if sel.numel() > 0:
            pr = F.normalize(self.prototypes.float(), dim=1)
            if self.proto_pool == "pooled":
                sim = F.normalize(self.proto_ln(feat), dim=-1) @ pr.T                       # (n_sel, P)
            else:
                st = F.normalize(self.proto_ln(tokens), dim=-1) @ pr.T                      # (n_sel, hw, P)
                st = torch.where(pvalid[..., None], st, torch.full_like(st, -2.0))
                sim = st.amax(1)
                sim = torch.where(pvalid.any(1, keepdim=True), sim, self.proto_null[None].expand_as(sim))
            raw = raw.index_copy(0, sel, sim.to(raw.dtype))
        return raw * torch.exp(self.proto_log_scale), raw

    def proto_diversity(self) -> torch.Tensor:
        """Mean squared off-diagonal cosine between prototypes (orthogonality penalty)."""
        pr = F.normalize(self.prototypes.float(), dim=1)
        G = pr @ pr.T
        P = G.shape[0]
        return ((G - torch.eye(P, device=G.device)) ** 2).sum() / max(P * (P - 1), 1)

    def _lesion_backbone_embed(self, x: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
        """Masked lesion crop (binarised detached lesion mask, square box +margin, every pixel outside the region
        set to 0, resized to lesion_crop_size) -> SAME backbone (second pass) -> [CLS, mean tokens] -> LN-Linear-tanh.
        Only pixels inside the predicted lesion mask (and the mask itself) influence the result."""
        from torchvision.ops import roi_align
        N = x.shape[0]
        H, W = x.shape[-2:]
        Lb = (L.detach().float() > self.lesion_thr).float()
        empty = Lb.flatten(1).mean(-1) < self.lesion_min_area
        out = torch.tanh(self.lesion_bb_null)[None].expand(N, -1).clone()
        sel = torch.nonzero(~empty, as_tuple=True)[0]
        if sel.numel() == 0:
            return out
        S = self.lesion_crop_size
        bx = square_boxes(Lb[sel], (H, W), self.lesion_margin)
        xc = roi_align(x.float(), torch.cat([sel.float()[:, None], bx], 1), (S, S), 1.0, 0, True)
        own = torch.arange(len(sel), device=x.device, dtype=torch.float32)[:, None]
        mc = roi_align(Lb[sel][:, None], torch.cat([own, bx], 1), (S, S), Lb.shape[-1] / W, 0, True)
        xin = xc * (mc >= 0.999).float()
        f2 = self.backbone(xin.to(x.dtype))                                  # second pass, shared weights
        pooled = f2.float().mean((2, 3))
        cls = getattr(self.backbone, "last_cls", None)
        feat = torch.cat([cls.float(), pooled], -1) if (cls is not None and getattr(self.backbone, "n_prefix", 0)) \
            else pooled
        return out.index_copy(0, sel, self.lesion_bb_head(feat).to(out.dtype))

    def check_mask_only_path(self, x: torch.Tensor) -> float | None:
        """Crop mode: max |d(logits, concept logits) / dF| with every mask detached. Must be exactly 0 (the masks
        are the only route from the global backbone to the diagnosis head). Returns None for other modes."""
        if not self.crop_mode:
            return None
        was = self.training
        self.eval()
        with torch.enable_grad():
            f = self.features(x).detach().float().requires_grad_(True)
            out = self.forward_heads(f, x, _detach_masks_all=True)
            tgt = out["logits"].float().sum() + out["concept_logits"].float().sum()
            gr = torch.autograd.grad(tgt, f, allow_unused=True)[0] if tgt.requires_grad else None
        self.train(was)
        return 0.0 if gr is None else float(gr.abs().max())

    def head_forward(self, hin: torch.Tensor) -> torch.Tensor:
        """Diagnosis head on a (possibly edited) head-input matrix (evaluator: ladder / sanity)."""
        return self.diag(hin.float())

    def _concept_repr(self, p, area, app, emb):
        """Per-concept representation that reaches the diagnosis head (for OIS): (N,K,r)."""
        if self.dh in ("global",) and emb is not None:
            return emb
        parts = [p[..., None]]
        if self.dh in ("hard", "soft_rr") and area is not None:
            parts.append(area.clamp_min(0).sqrt()[..., None])
        if self.dh == "soft_rr" and app is not None:
            parts.append(app)
        return torch.cat(parts, -1)
