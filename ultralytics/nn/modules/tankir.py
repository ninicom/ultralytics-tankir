# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""TankIR modules for thermal tank detection (fork of Ultralytics 8.4.218).

Usage: from ultralytics.nn.modules import tankir as T; T.build("rcs_zero,pis_zero", nc=1, weights="yolo11s.pt")
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MambaBlock(nn.Module):
    """Channel-preserving 2D Mamba-style block (forward + backward scan over the flattened feature map)."""

    def __init__(self, c, d_state=16):
        super().__init__()
        self.c, self.n = c, d_state
        self.norm = nn.LayerNorm(c)
        self.in_proj = nn.Linear(c, 2 * c)
        self.conv = nn.Conv1d(c, c, 3, padding=1, groups=c)
        self.dt_proj = nn.Linear(c, c)
        self.B_proj = nn.Linear(c, d_state, bias=False)
        self.C_proj = nn.Linear(c, d_state, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(c, 1)))
        self.D = nn.Parameter(torch.ones(c))
        self.out_proj = nn.Linear(c, c)
        self.gamma = nn.Parameter(torch.zeros(1))  # starts as identity -> keeps pretrained behaviour at init
        nn.init.normal_(self.dt_proj.weight, std=1e-3)
        nn.init.constant_(self.dt_proj.bias, math.log(math.expm1(0.05)))

    def _scan(self, u, dt, Bm, Cm):
        A = -torch.exp(self.A_log)                                  # (c, n)
        dA = torch.exp(dt.unsqueeze(-1) * A)                         # (b, L, c, n)
        dBu = dt.unsqueeze(-1) * Bm.unsqueeze(2) * u.unsqueeze(-1)   # (b, L, c, n)
        h = u.new_zeros(u.shape[0], self.c, self.n)
        ys = []
        for t in range(u.shape[1]):
            h = dA[:, t] * h + dBu[:, t]
            ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))
        return torch.stack(ys, 1) + u * self.D

    def forward(self, x):
        b, c, hh, ww = x.shape
        s = x.flatten(2).transpose(1, 2)                             # (b, L, c)
        u, g = self.in_proj(self.norm(s)).chunk(2, dim=-1)
        u = F.silu(self.conv(u.transpose(1, 2)).transpose(1, 2))
        with torch.autocast(device_type='cuda', enabled=False):
            u32 = u.float()
            dt = F.softplus(self.dt_proj(u32))
            Bm, Cm = self.B_proj(u32), self.C_proj(u32)
            y = self._scan(u32, dt, Bm, Cm)
            y = y + self._scan(u32.flip(1), dt.flip(1), Bm.flip(1), Cm.flip(1)).flip(1)
        y = self.out_proj((y.to(u.dtype)) * F.silu(g))
        s = s + self.gamma * y
        return s.transpose(1, 2).reshape(b, c, hh, ww)


class RepViTBlock(nn.Module):
    """RepViT-style block: (dw3x3-BN + dw1x1-BN + identity) token mixer, SE, then 1x1 FFN with residual."""

    def __init__(self, c, ratio=2, se=True):
        super().__init__()
        self.dw3 = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False), nn.BatchNorm2d(c))
        self.dw1 = nn.Sequential(nn.Conv2d(c, c, 1, 1, 0, groups=c, bias=False), nn.BatchNorm2d(c))
        self.bn = nn.BatchNorm2d(c)
        r = max(8, c // 4)
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, r, 1), nn.ReLU(inplace=True), nn.Conv2d(r, c, 1), nn.Sigmoid()) if se else None
        self.ffn = nn.Sequential(nn.Conv2d(c, c * ratio, 1, bias=False), nn.BatchNorm2d(c * ratio), nn.GELU(),
                                 nn.Conv2d(c * ratio, c, 1, bias=False), nn.BatchNorm2d(c))

    def forward(self, x):
        t = self.dw3(x) + self.dw1(x) + self.bn(x)
        if self.se is not None:
            t = t * self.se(t)
        return t + self.ffn(t)


# ------------------------------------------------------------------ TankIR-YOLO modules
class StripAttention(nn.Module):
    """M1 – Barrel-aware Strip Attention (BSA).

    Long horizontal / vertical depthwise strip convolutions plus row/column strip pooling,
    to emphasise elongated structures (gun barrel, hull/track line) with global context along
    each axis. Only pooling + convs -> TensorRT friendly. Residual gate starts at 0 (identity).
    """

    def __init__(self, c, k=11):
        super().__init__()
        self.h_conv = nn.Conv2d(c, c, (1, k), padding=(0, k // 2), groups=c, bias=False)
        self.v_conv = nn.Conv2d(c, c, (k, 1), padding=(k // 2, 0), groups=c, bias=False)
        self.row = nn.Conv1d(c, c, 7, padding=3, groups=c, bias=False)   # along H (pooled over W)
        self.col = nn.Conv1d(c, c, 7, padding=3, groups=c, bias=False)   # along W (pooled over H)
        self.fuse = nn.Sequential(nn.Conv2d(c, c, 1, bias=False), nn.BatchNorm2d(c))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        rp = self.row(x.mean(3)).unsqueeze(3)          # (b,c,h,1)
        cp = self.col(x.mean(2)).unsqueeze(2)          # (b,c,1,w)
        a = torch.sigmoid(self.fuse(self.h_conv(x) + self.v_conv(x) + rp + cp))
        return x + self.gamma * (x * a)


class WithPost(nn.Module):
    """Run an existing YOLO layer, then a channel-preserving post module (keeps pretrained weights)."""

    def __init__(self, base, post):
        super().__init__()
        self.base, self.post = base, post

    def forward(self, x):
        return self.post(self.base(x))


class PolarityStem(nn.Module):
    """M3 – Polarity-invariant thermal stem: shared first conv on x and its inversion (1-x), averaged.

    Inverting the input (white-hot <-> black-hot) swaps the two branches, so the stem output is
    exactly invariant to thermal polarity.
    """

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, x):
        return 0.5 * (self.base(x) + self.base(1.0 - x))


class RCSBlock(nn.Module):
    """C3k2-RCS bottleneck – Reparameterizable Cross-Strip block.

    Train: parallel depthwise 3x3, 1xk, kx1 (+BN each, + identity BN) -> SE -> 1x1 FFN, residual.
    Inference: call .reparam() to merge the three depthwise branches + identity into ONE kxk depthwise conv.
    """

    def __init__(self, c, k=7, ratio=2):
        super().__init__()
        self.c, self.k = c, k
        self.d3 = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1, groups=c, bias=False), nn.BatchNorm2d(c))
        self.dh = nn.Sequential(nn.Conv2d(c, c, (1, k), 1, (0, k // 2), groups=c, bias=False), nn.BatchNorm2d(c))
        self.dv = nn.Sequential(nn.Conv2d(c, c, (k, 1), 1, (k // 2, 0), groups=c, bias=False), nn.BatchNorm2d(c))
        self.idn = nn.BatchNorm2d(c)
        self.merged = None
        r = max(8, c // 4)
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, r, 1), nn.ReLU(inplace=True), nn.Conv2d(r, c, 1), nn.Sigmoid())
        self.ffn = nn.Sequential(nn.Conv2d(c, c * ratio, 1, bias=False), nn.BatchNorm2d(c * ratio), nn.GELU(),
                                 nn.Conv2d(c * ratio, c, 1, bias=False), nn.BatchNorm2d(c))

    def _mix(self, x):
        if self.merged is not None:
            return self.merged(x)
        return self.d3(x) + self.dh(x) + self.dv(x) + self.idn(x)

    def forward(self, x):
        t = self._mix(x)
        t = t * self.se(t)
        return t + self.ffn(t)

    @staticmethod
    def _fuse(conv_w, bn):
        std = (bn.running_var + bn.eps).sqrt()
        return conv_w * (bn.weight / std).reshape(-1, 1, 1, 1), bn.bias - bn.running_mean * bn.weight / std

    @torch.no_grad()
    def reparam(self):
        """Merge 3x3 + 1xk + kx1 + identity depthwise branches into one kxk depthwise conv (exact, eval mode)."""
        if self.merged is not None:
            return
        k, c, p = self.k, self.c, self.k // 2
        W = torch.zeros(c, 1, k, k, device=self.d3[0].weight.device)
        B = torch.zeros(c, device=W.device)
        w, b = self._fuse(self.d3[0].weight, self.d3[1]); W[:, :, p - 1:p + 2, p - 1:p + 2] += w; B += b
        w, b = self._fuse(self.dh[0].weight, self.dh[1]); W[:, :, p:p + 1, :] += w; B += b
        w, b = self._fuse(self.dv[0].weight, self.dv[1]); W[:, :, :, p:p + 1] += w; B += b
        idw = torch.zeros(c, 1, k, k, device=W.device); idw[:, 0, p, p] = 1.0
        w, b = self._fuse(idw, self.idn); W += w; B += b
        self.merged = nn.Conv2d(c, c, k, 1, p, groups=c, bias=True).to(W.device)
        self.merged.weight.copy_(W); self.merged.bias.copy_(B)
        for n in ('d3', 'dh', 'dv', 'idn'):
            delattr(self, n)


class TCDown(nn.Module):
    """Thermal-Contrast Downsampling: pretrained stride-2 Conv + gated max-pool path that keeps hot-spot peaks."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        c1, c2 = base.conv.in_channels, base.conv.out_channels
        self.peak = nn.Sequential(nn.MaxPool2d(2, 2), nn.Conv2d(c1, c2, 1, bias=False), nn.BatchNorm2d(c2))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        y = self.base(x)
        z = self.peak(x)
        if z.shape[-2:] != y.shape[-2:]:
            z = F.interpolate(z, size=y.shape[-2:], mode='nearest')
        return y + self.gamma * z


class ASC(nn.Module):
    """Axial Strip Context (replaces C2PSA at P5): global row/column context in O(H+W), TensorRT-friendly."""

    def __init__(self, c, k=7, ratio=2):
        super().__init__()
        self.pre = nn.Sequential(nn.Conv2d(c, c, 1, bias=False), nn.BatchNorm2d(c), nn.SiLU())
        self.row = nn.Conv1d(c, c, k, padding=k // 2, groups=c)
        self.col = nn.Conv1d(c, c, k, padding=k // 2, groups=c)
        self.mix = nn.Sequential(nn.Conv2d(c, c, 1, bias=False), nn.BatchNorm2d(c))
        self.ffn = nn.Sequential(nn.Conv2d(c, c * ratio, 1, bias=False), nn.BatchNorm2d(c * ratio), nn.SiLU(),
                                 nn.Conv2d(c * ratio, c, 1, bias=False), nn.BatchNorm2d(c))

    def forward(self, x):
        t = self.pre(x)
        a = torch.sigmoid(self.mix(self.row(t.mean(3)).unsqueeze(3) + self.col(t.mean(2)).unsqueeze(2)))
        x = x + t * a
        return x + self.ffn(x)


class ZeroRes(nn.Module):
    """Y = base(X) + alpha * branch(X), alpha = 0 at init -> identical to the pretrained model at epoch 0."""

    def __init__(self, base, branch):
        super().__init__()
        self.base, self.branch = base, branch
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.base(x) + self.alpha * self.branch(x)


class PolarityStemZero(nn.Module):
    """Zero-init PIS: f(x) + alpha * 0.5 * (f(1-x) - f(x)); alpha=0 -> pretrained stem, alpha=1 -> exact PIS."""

    def __init__(self, base):
        super().__init__()
        self.base = base
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        a = self.base(x)
        return a + self.alpha * 0.5 * (self.base(1.0 - x) - a)


def apply_rcs_zero(model, layers=(2, 4, 6, 8), k=7):
    """Keep the pretrained C3k2 bottlenecks and add RCS as a zero-initialised residual branch."""
    for i in layers:
        m = model.model[i]
        m.m = nn.ModuleList(ZeroRes(b, RCSBlock(m.c, k=k)) for b in m.m)
    return model


def apply_m3_zero(model):
    _wrap(model, 0, PolarityStemZero(model.model[0]))
    return model


def apply_rcs(model, layers=(2, 4, 6, 8), k=7):
    for i in layers:
        m = model.model[i]
        m.m = nn.ModuleList(RCSBlock(m.c, k=k) for _ in range(len(m.m)))
    return model


def apply_tcdown(model, layers=(1, 3, 5, 7)):
    for i in layers:
        _wrap(model, i, TCDown(model.model[i]))
    return model


def apply_asc(model, idx=10):
    c = model.model[idx].cv2.conv.out_channels
    _wrap(model, idx, ASC(c))
    return model


def reparam_all(model):
    for m in model.modules():
        if isinstance(m, RCSBlock):
            m.reparam()
    return model


def _wrap(model, idx, new):
    old = model.model[idx]
    for a in ('f', 'i', 'type'):
        if hasattr(old, a):
            setattr(new, a, getattr(old, a))
    new.np = sum(p.numel() for p in new.parameters())
    model.model[idx] = new


def apply_m1(model, layers=(25, 28)):
    """Add StripAttention after the P4 / P5 neck outputs of the P2-P5 model."""
    for i in layers:
        base = model.model[i]
        _wrap(model, i, WithPost(base, StripAttention(base.cv2.conv.out_channels)))
    return model


def apply_m3(model):
    _wrap(model, 0, PolarityStem(model.model[0]))
    return model


def _swap(model, idx, new):
    old = model.model[idx]
    for a in ('f', 'i', 'type', 'np'):
        if hasattr(old, a):
            setattr(new, a, getattr(old, a))
    new.np = sum(p.numel() for p in new.parameters())
    model.model[idx] = new


def apply_cm_style(model):
    """Replace C2PSA (layer 10, P5) with a MambaBlock of the same width."""
    c = model.model[10].cv2.conv.out_channels
    _swap(model, 10, MambaBlock(c))
    return model


def apply_atf_style(model, backbone_layers=(2, 4, 6, 8)):
    """Replace the bottlenecks inside backbone C3k2 stages with RepViT blocks (channel-preserving)."""
    for i in backbone_layers:
        m = model.model[i]
        m.m = nn.ModuleList(RepViTBlock(m.c) for _ in range(len(m.m)))
    return model


YOLO11_P2_YAML = """
nc: 1
scales:
  s: [0.50, 0.50, 1024]
backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [512, False, 0.25]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 2, C3k2, [512, True]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 2, C3k2, [1024, True]]
  - [-1, 1, SPPF, [1024, 5]]
  - [-1, 2, C2PSA, [1024]]
head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 2], 1, Concat, [1]]
  - [-1, 2, C3k2, [128, False]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [[-1, 16], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, False]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]]
  - [-1, 2, C3k2, [512, False]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]]
  - [-1, 2, C3k2, [1024, True]]
  - [[19, 22, 25, 28], 1, Detect, [nc]]
"""

def build(use: str = "", nc: int = 1, weights: str | None = "yolo11s.pt", p2: bool = True):
    """Build a YOLO11s(-P2) DetectionModel with TankIR blocks.

    Args:
        use (str): comma-separated blocks: rcs, rcs_zero, pis, pis_zero, bsa (strip attention P4/P5), tcd, asc.
        nc (int): number of classes.
        weights (str | None): pretrained checkpoint to load before inserting blocks (keeps pretrained features).
        p2 (bool): use the P2-P5 head (TankIR baseline A0) instead of the standard P3-P5 head.
    """
    import tempfile, os
    import torch
    from ultralytics.nn.tasks import DetectionModel

    if p2:
        f = os.path.join(tempfile.gettempdir(), "yolo11s-p2.yaml")
        with open(f, "w") as fh:
            fh.write(YOLO11_P2_YAML)
        cfg = f
    else:
        cfg = "yolo11s.yaml"
    m = DetectionModel(cfg, nc=nc, verbose=False)
    if weights:
        ck = torch.load(weights, map_location="cpu", weights_only=False) if isinstance(weights, str) else weights
        m.load(ck)
    u = {x.strip() for x in use.split(",") if x.strip()}
    if "asc" in u: apply_asc(m)
    if "tcd" in u: apply_tcdown(m)
    if "rcs" in u: apply_rcs(m)
    if "rcs_zero" in u: apply_rcs_zero(m)
    if "bsa" in u: apply_m1(m)
    if "pis" in u: apply_m3(m)
    if "pis_zero" in u: apply_m3_zero(m)
    return m
