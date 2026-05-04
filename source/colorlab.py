"""Color-space conversions, CIEDE2000, MDN helpers, chroma diagnostics.

The differentiable Lab->RGB is used inside the training graph (LPIPS, adversarial
loss). The NumPy version is for evaluation and visualization only -- it goes
through scikit-image and is more accurate but breaks autograd.
"""

import math
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from skimage import color as skcolor
from skimage.color import deltaE_ciede2000


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(0, 1) * 255).to(torch.uint8)


def lab_to_rgb_diff(L: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
    """Differentiable Lab -> sRGB. Used in the G backward pass (LPIPS, adv).

    Inputs are normalized: L in [0,1], a/b in [-1,1] (i.e. the network outputs).
    """
    Ls = L * 100
    a, b = ab[:, 0:1] * 128, ab[:, 1:2] * 128
    fy = (Ls + 16) / 116
    fx = a / 500 + fy
    fz = fy - b / 200
    d = 6 / 29

    def fi(t):
        return torch.where(t > d, t ** 3, 3 * d ** 2 * (t - 4 / 29))

    X, Y, Z = 0.950456 * fi(fx), fi(fy), 1.088754 * fi(fz)
    R = 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
    G = -0.969266 * X + 1.876011 * Y + 0.041556 * Z
    B = 0.055643 * X - 0.204026 * Y + 1.057225 * Z

    def sr(c):
        return torch.where(c > 0.0031308,
                           1.055 * c.clamp(min=1e-8).pow(1 / 2.4) - 0.055,
                           12.92 * c)

    return torch.cat([sr(R), sr(G), sr(B)], 1).clamp(0, 1)


def lab_to_rgb_np(L: torch.Tensor, ab: torch.Tensor) -> torch.Tensor:
    """NumPy / scikit-image Lab -> sRGB. Eval and visualization only."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Ln = (L.detach().cpu().numpy().transpose(0, 2, 3, 1) * 100).astype(np.float32)
        An = (ab.detach().cpu().numpy().transpose(0, 2, 3, 1) * 128).astype(np.float32)
        rgb = [np.clip(skcolor.lab2rgb(np.concatenate([Ln[i], An[i]], -1)), 0, 1)
               for i in range(Ln.shape[0])]
    return torch.from_numpy(np.stack(rgb)).permute(0, 3, 1, 2).float().to(L.device)


def ciede2000_batch(Lg: torch.Tensor, ag: torch.Tensor, ap: torch.Tensor) -> float:
    Ln = Lg.detach().cpu().numpy().transpose(0, 2, 3, 1) * 100
    A1 = ag.detach().cpu().numpy().transpose(0, 2, 3, 1) * 128
    A2 = ap.detach().cpu().numpy().transpose(0, 2, 3, 1) * 128
    return float(np.mean([
        deltaE_ciede2000(np.concatenate([Ln[i], A1[i]], -1),
                         np.concatenate([Ln[i], A2[i]], -1)).mean()
        for i in range(Ln.shape[0])
    ]))


# ── MDN helpers ─────────────────────────────────────────────────────────────

def mdn_nll(pi, mu, log_sigma, target, min_sigma=1e-3,
            log_sigma_min=-7.0, log_sigma_max=3.0):
    B, K, H, W = pi.shape
    x = target.unsqueeze(1).expand(-1, K, -1, -1, -1)
    ls = log_sigma.clamp(log_sigma_min, log_sigma_max)
    s = torch.exp(ls).clamp(min=min_sigma)
    lp = -0.5 * (((x - mu) / s) ** 2 + 2 * torch.log(s) + math.log(2 * math.pi))
    return -torch.logsumexp(torch.log_softmax(pi, 1) + lp.sum(2), 1).mean()


@torch.no_grad()
def mdn_expected_ab(pi, mu):
    """E[ab] = sum_k pi_k * mu_k."""
    return (torch.softmax(pi, 1).unsqueeze(2) * mu).sum(1)


@torch.no_grad()
def mdn_map_ab(pi, mu):
    """MAP: pick mu of the highest-weight component at each pixel."""
    k = torch.softmax(pi, 1).argmax(1, keepdim=True).unsqueeze(2).expand(-1, -1, 2, -1, -1)
    return mu.gather(1, k).squeeze(1)


@torch.no_grad()
def mdn_sample_ab(pi, mu, log_sigma, temperature: float = 1.0, min_sigma: float = 1e-3):
    w = torch.softmax(pi / temperature, 1)
    B, K, _, H, W = mu.shape
    idx = (torch.multinomial(w.permute(0, 2, 3, 1).reshape(-1, K), 1)
                .view(B, H, W, 1, 1).permute(0, 3, 4, 1, 2)
                .expand(-1, -1, 2, -1, -1))
    m = mu.gather(1, idx).squeeze(1)
    ls = log_sigma.gather(1, idx).squeeze(1)
    return m + torch.randn_like(m) * torch.exp(ls).clamp(min=min_sigma) * temperature


@torch.no_grad()
def mdn_best_of_n(pi, mu, log_sigma, L, ab_gt, n: int = 5, min_sigma: float = 1e-3):
    """Sample N times, return the sample with the lowest L1 vs GT."""
    best, best_l1 = None, float('inf')
    for _ in range(n):
        s = mdn_sample_ab(pi, mu, log_sigma, 1.0, min_sigma=min_sigma)
        l1 = F.l1_loss(s, ab_gt).item()
        if l1 < best_l1:
            best_l1, best = l1, s
    return best


# ── Chroma diagnostics / loss ──────────────────────────────────────────────

def chroma_diagnostics(ab_gt, ab_pred):
    gt = ab_gt.detach().cpu().numpy().transpose(0, 2, 3, 1) * 128
    pr = ab_pred.detach().cpu().numpy().transpose(0, 2, 3, 1) * 128
    c_gt = np.sqrt(gt[..., 0] ** 2 + gt[..., 1] ** 2).ravel()
    c_pr = np.sqrt(pr[..., 0] ** 2 + pr[..., 1] ** 2).ravel()
    return {
        "a_gt": gt[..., 0].ravel(), "b_gt": gt[..., 1].ravel(),
        "a_pr": pr[..., 0].ravel(), "b_pr": pr[..., 1].ravel(),
        "chroma_gt": c_gt, "chroma_pr": c_pr,
        "chroma_ratio": np.mean(c_pr) / max(np.mean(c_gt), 1e-6),
    }


def chroma_loss(ab_pred, ab_gt):
    """L_chroma = L1(C_pred, C_gt) where C = sqrt(a^2 + b^2)."""
    C_pred = torch.sqrt(ab_pred[:, 0:1] ** 2 + ab_pred[:, 1:2] ** 2 + 1e-8)
    C_gt   = torch.sqrt(ab_gt[:, 0:1]   ** 2 + ab_gt[:, 1:2]   ** 2 + 1e-8)
    return F.l1_loss(C_pred, C_gt)
