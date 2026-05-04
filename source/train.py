"""Training entry point: runs the four ablation experiments end-to-end.

This is the CLI equivalent of `colab_notebook.ipynb`. It:

  1. Makes sure the dataset is present (or downloads it via kagglehub).
  2. Builds train/val/test splits and caches them on disk.
  3. Computes BLIP captions + CLIP token embeddings for each split.
  4. Trains four models: baseline, ours, ours+chroma, ours+MDN.
  5. Saves per-experiment checkpoints + metrics + qualitative grids.
  6. Writes summary tables and the panel/comparison figures used for
     mid-experiment inspection.

Run:    python -m source.train
        python -m source.train --n-images 5000 --epochs 10
        IMAGES_DIR=/path/to/images python -m source.train
"""

import argparse
import copy
import json
import os

import lpips
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage import color as skcolor
from torchmetrics.image import (
    PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure,
)
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm.auto import tqdm

from .colorlab import (
    chroma_diagnostics, chroma_loss, ciede2000_batch,
    lab_to_rgb_diff, lab_to_rgb_np,
    mdn_best_of_n, mdn_expected_ab, mdn_map_ab, mdn_nll, mdn_sample_ab,
    to_uint8,
)
from .config import default_config, seed_all
from .data import (
    list_images, make_or_load_split,
    make_plain_loaders, make_text_loaders, unpack_batch,
)
from .dataset_setup import ensure_images
from .models import PatchGAN, build_generator
from .text_cache import build_text_cache


# ── Validation / training helpers ───────────────────────────────────────────

def val_epoch(G, mode, use_text, loader, lpips_fn, dev, cfg):
    """Validate on val set. Eval-only — uses NumPy Lab->RGB."""
    G.eval()
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(dev)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(dev)
    l1s, lps, des, cls, nlls = [], [], [], [], []
    with torch.no_grad():
        for b in loader:
            L, ab, rgb, tk, mk = unpack_batch(b, use_text, dev)
            out = G(L, text_tokens=tk, text_mask=mk, use_text=use_text)
            ap = out["ab"] if mode == "reg" else mdn_expected_ab(out["pi"], out["mu_mdn"])
            l1s.append(F.l1_loss(ap, ab).item())
            cls.append(chroma_loss(ap, ab).item())
            if mode == "mdn":
                nlls.append(mdn_nll(
                    out["pi"], out["mu_mdn"], out["log_sigma"], ab,
                    cfg["mdn_min_sigma"], cfg["mdn_log_sigma_min"], cfg["mdn_log_sigma_max"],
                ).item())
            rp = lab_to_rgb_np(L, ap).clamp(0, 1)
            if torch.isnan(rp).any():
                continue
            psnr.update(rp, rgb); ssim.update(rp, rgb)
            lps.append(lpips_fn(rp * 2 - 1, rgb * 2 - 1).mean().item())
            des.append(ciede2000_batch(L, ab, ap))

    r = {
        "val_l1":     float(np.mean(l1s)) if l1s else 999.0,
        "val_psnr":   float(psnr.compute().item()),
        "val_ssim":   float(ssim.compute().item()),
        "val_lpips":  float(np.mean(lps)) if lps else 999.0,
        "val_deltae": float(np.mean(des)) if des else 999.0,
        "val_chroma": float(np.mean(cls)) if cls else 999.0,
    }
    if mode == "mdn":
        r["val_nll"] = float(np.mean(nlls)) if nlls else 999.0
    return r


def _save_qualitative_grid(G, te_loader, mode, use_text, dev, name, out_dir):
    G.eval()
    try:
        b = next(iter(te_loader))
        L, ab, rgb, tk, mk = unpack_batch(b, use_text, dev)
        L, ab, rgb = L[:6], ab[:6], rgb[:6]
        if tk is not None:
            tk, mk = tk[:6], mk[:6]
        with torch.no_grad():
            out = G(L, text_tokens=tk if use_text else None,
                    text_mask=mk if use_text else None, use_text=use_text)
            ap = out["ab"] if mode == "reg" else mdn_expected_ab(out["pi"], out["mu_mdn"])
            rp = lab_to_rgb_np(L, ap).clamp(0, 1)

        n = min(6, L.size(0))
        fig, ax = plt.subplots(n, 3, figsize=(9, 3 * n), squeeze=False)
        for i in range(n):
            ax[i, 0].imshow(L[i, 0].cpu().numpy(), cmap="gray"); ax[i, 0].axis("off"); ax[i, 0].set_title("L")
            ax[i, 1].imshow(rgb[i].permute(1, 2, 0).cpu().numpy()); ax[i, 1].axis("off"); ax[i, 1].set_title("GT")
            ax[i, 2].imshow(rp[i].permute(1, 2, 0).cpu().numpy()); ax[i, 2].axis("off"); ax[i, 2].set_title(name)
        plt.suptitle(f"Qualitative: {name}"); plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "qualitative_grid.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        return ab, ap
    except Exception as e:
        print(f"  qualitative grid skipped: {e}")
        return None, None


def _save_chroma_histogram(ab, ap, name, out_dir):
    if ab is None or ap is None:
        return
    try:
        d = chroma_diagnostics(ab, ap)
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].hist(d["a_gt"], bins=60, alpha=.5, label="GT", density=True)
        ax[0].hist(d["a_pr"], bins=60, alpha=.5, label="Pred", density=True)
        ax[0].set_title("a channel"); ax[0].legend()
        ax[1].hist(d["b_gt"], bins=60, alpha=.5, label="GT", density=True)
        ax[1].hist(d["b_pr"], bins=60, alpha=.5, label="Pred", density=True)
        ax[1].set_title("b channel"); ax[1].legend()
        ax[2].hist(d["chroma_gt"], bins=60, alpha=.5, label="GT", density=True)
        ax[2].hist(d["chroma_pr"], bins=60, alpha=.5, label="Pred", density=True)
        ax[2].set_title(f"Chroma C (ratio={d['chroma_ratio']:.3f})"); ax[2].legend()
        plt.suptitle(f"Chroma Histogram: {name}"); plt.tight_layout()
        fig.savefig(os.path.join(out_dir, "chroma_histogram.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"  chroma histogram skipped: {e}")


def run_experiment(name, mode, use_text, text_mode,
                   train_loader, val_loader, test_loader,
                   dev, cfg, lambda_chroma: float = 0.0):
    """Train one configuration end-to-end and return a dict of test metrics."""
    seed_all(cfg["seed"])
    print(f"\n{'='*60}")
    print(f"  {name} | mode={mode} | text_mode={text_mode} | "
          f"lam_chroma={lambda_chroma}")
    print(f"{'='*60}")

    run_dir = os.path.join(cfg["save_dir"], name)
    os.makedirs(run_dir, exist_ok=True)

    # Snapshot the experiment config for reproducibility.
    cfg_snapshot = {
        k: (v if isinstance(v, (int, float, bool, str, list, dict, type(None))) else str(v))
        for k, v in cfg.items()
    }
    cfg_snapshot.update({
        "experiment": name, "mode": mode, "text_mode": text_mode,
        "use_text": use_text, "text_in_dim": 512,
        "lambda_chroma": lambda_chroma,
    })
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(cfg_snapshot, f, indent=2)

    G = build_generator(use_text=use_text, mode=mode, text_mode=text_mode,
                        K=cfg["mdn_K"], text_in_dim=512).to(dev)
    D = PatchGAN().to(dev)

    lr_g = cfg["lr_g_mdn"] if mode == "mdn" else cfg["lr_g_reg"]
    oG = torch.optim.Adam(G.parameters(), lr=lr_g, betas=(.5, .999))
    oD = torch.optim.Adam(D.parameters(), lr=cfg["lr_d"], betas=(.5, .999))
    sG = torch.optim.lr_scheduler.CosineAnnealingLR(oG, cfg["epochs"])
    sD = torch.optim.lr_scheduler.CosineAnnealingLR(oD, cfg["epochs"])
    bce = nn.BCEWithLogitsLoss()
    lpips_fn = lpips.LPIPS(net='alex').to(dev)

    la   = cfg["lambda_adv_mdn"]   if mode == "mdn" else cfg["lambda_adv_reg"]
    lr_c = cfg["lambda_recon_mdn"] if mode == "mdn" else cfg["lambda_recon_reg"]

    hist, best_val, best_state, patience = [], float('inf'), None, 0

    for ep in range(cfg["epochs"]):
        G.train(); D.train()
        warmup_key = "gan_warmup_epochs_mdn" if mode == "mdn" else "gan_warmup_epochs_reg"
        gan_on = ep >= cfg.get(warmup_key, 0)

        sum_rc = sum_adv = sum_lp = sum_mdn = sum_chroma = 0.0
        nb = 0

        for batch in tqdm(train_loader, desc=f"{name} ep{ep+1}", leave=False):
            L, ab, rgb, tk, mk = unpack_batch(batch, use_text, dev)

            # ── D step ──
            if gan_on:
                oD.zero_grad(set_to_none=True)
                with torch.no_grad():
                    od = G(L, text_tokens=tk, text_mask=mk, use_text=use_text)
                    af = od["ab"] if mode == "reg" else mdn_expected_ab(od["pi"], od["mu_mdn"])
                    rf = lab_to_rgb_diff(L, af)  # differentiable Lab->RGB
                d_real = D(L, rgb)
                d_fake = D(L, rf)
                ld = .5 * (
                    bce(d_real, torch.ones_like(d_real) * .9) +
                    bce(d_fake, torch.zeros_like(d_fake))
                )
                if torch.isfinite(ld):
                    ld.backward(); oD.step()
                else:
                    continue

            # ── G step ──
            oG.zero_grad(set_to_none=True)
            out = G(L, text_tokens=tk, text_mask=mk, use_text=use_text)
            if mode == "reg":
                ap = out["ab"]; rc = F.l1_loss(ap, ab); mdn_l = torch.tensor(0.0, device=dev)
            else:
                mdn_l = mdn_nll(out["pi"], out["mu_mdn"], out["log_sigma"], ab,
                                cfg["mdn_min_sigma"], cfg["mdn_log_sigma_min"], cfg["mdn_log_sigma_max"])
                ap = mdn_expected_ab(out["pi"], out["mu_mdn"]); rc = mdn_l

            if torch.isnan(ap).any():
                continue

            rp = lab_to_rgb_diff(L, ap)  # differentiable for LPIPS + adv
            d_pred = D(L, rp)
            adv = bce(d_pred, torch.ones_like(d_pred)) if gan_on else torch.tensor(0., device=dev)
            lp  = lpips_fn(rp * 2 - 1, rgb * 2 - 1).mean() if cfg["lambda_lpips"] > 0 else torch.tensor(0., device=dev)
            lc  = chroma_loss(ap, ab) if lambda_chroma > 0 else torch.tensor(0., device=dev)
            lG  = lr_c * rc + la * adv + cfg["lambda_lpips"] * lp + lambda_chroma * lc

            if not torch.isfinite(lG):
                continue
            lG.backward()
            nn.utils.clip_grad_norm_(G.parameters(), cfg["clip_grad"])
            oG.step()

            sum_rc += rc.item(); sum_adv += adv.item(); sum_lp += lp.item()
            sum_mdn += mdn_l.item() if mode == "mdn" else 0.0
            sum_chroma += lc.item(); nb += 1

        sG.step(); sD.step()

        n = max(nb, 1)
        loss_mag = {
            "recon":   sum_rc / n, "adv": sum_adv / n,
            "lpips":   sum_lp / n, "mdn_nll": sum_mdn / n,
            "chroma":  sum_chroma / n,
        }
        vm = val_epoch(G, mode, use_text, val_loader, lpips_fn, dev, cfg)
        row = {"epoch": ep + 1,
               **{f"train_{k}": v for k, v in loss_mag.items()}, **vm}
        hist.append(row)

        print(f"  ep{ep+1} LOSS: rc={loss_mag['recon']:.4f} adv={loss_mag['adv']:.4f} "
              f"lpips={loss_mag['lpips']:.4f} mdn={loss_mag['mdn_nll']:.4f} "
              f"chroma={loss_mag['chroma']:.4f}")
        val_line = (f"         VAL:  psnr={vm['val_psnr']:.2f} ssim={vm['val_ssim']:.4f} "
                    f"lpips={vm['val_lpips']:.4f} dE={vm['val_deltae']:.2f} "
                    f"chroma={vm['val_chroma']:.4f}")
        if mode == "mdn":
            val_line += f" nll={vm.get('val_nll', 0):.4f}"
        print(val_line)

        # Best-checkpoint selection: val_nll for MDN, val_l1 for deterministic.
        if cfg.get("early_stopping"):
            ckpt_metric = vm.get("val_nll", 999) if mode == "mdn" else vm["val_l1"]
            if ckpt_metric < best_val - cfg.get("es_min_delta", 0):
                best_val = ckpt_metric
                best_state = copy.deepcopy(G.state_dict())
                patience = 0
                torch.save(best_state, os.path.join(run_dir, "best_val_checkpoint.pt"))
            else:
                patience += 1
            if patience >= cfg["es_patience"]:
                print(f"  ** Early stop ep{ep+1}")
                break

    if cfg.get("early_stopping") and best_state is not None:
        G.load_state_dict(best_state)
    torch.save(G.state_dict(), os.path.join(run_dir, "final_checkpoint.pt"))

    viz_bundle = {
        "experiment": name, "mode": mode, "use_text": use_text,
        "text_mode": text_mode, "text_in_dim": 512,
        "mdn_K": cfg["mdn_K"], "state_dict": G.state_dict(),
    }
    torch.save(viz_bundle, os.path.join(run_dir, "viz_model_bundle.pt"))
    pd.DataFrame(hist).to_csv(os.path.join(run_dir, "metrics_history.csv"), index=False)
    print(f"  Saved checkpoints: final_checkpoint.pt + viz_model_bundle.pt")

    # ── Final TEST set evaluation (used once per experiment) ──
    G.eval()
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(dev)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(dev)
    fid  = FrechetInceptionDistance(feature=2048).to(dev)
    lps, des, c_prs, c_gts = [], [], [], []
    with torch.no_grad():
        for bi, b in enumerate(tqdm(test_loader, desc=f"Test {name}")):
            if cfg.get("eval_max_batches") and bi >= cfg["eval_max_batches"]:
                break
            L, ab, rgb, tk, mk = unpack_batch(b, use_text, dev)
            out = G(L, text_tokens=tk, text_mask=mk, use_text=use_text)
            ap = out["ab"] if mode == "reg" else mdn_expected_ab(out["pi"], out["mu_mdn"])
            rp = lab_to_rgb_np(L, ap).clamp(0, 1)
            if torch.isnan(rp).any():
                continue
            psnr.update(rp, rgb); ssim.update(rp, rgb)
            fid.update(to_uint8(rgb), real=True)
            fid.update(to_uint8(rp),  real=False)
            lps.append(lpips_fn(rp * 2 - 1, rgb * 2 - 1).mean().item())
            des.append(ciede2000_batch(L, ab, ap))
            c_prs.append(torch.sqrt(ap[:, 0:1] ** 2 + ap[:, 1:2] ** 2 + 1e-8).mean().item())
            c_gts.append(torch.sqrt(ab[:, 0:1] ** 2 + ab[:, 1:2] ** 2 + 1e-8).mean().item())

    cr = np.mean(c_prs) / max(np.mean(c_gts), 1e-6) if c_gts else 0.0
    test_metrics = {
        "psnr":         float(psnr.compute().item()),
        "ssim":         float(ssim.compute().item()),
        "fid_2048":     float(fid.compute().item()),
        "lpips":        float(np.mean(lps)) if lps else 0.0,
        "delta_e":      float(np.mean(des)) if des else 0.0,
        "chroma_ratio": float(cr),
        "model": G,
    }
    test_serializable = {k: v for k, v in test_metrics.items() if k != "model"}
    with open(os.path.join(run_dir, "test_metrics.json"), "w") as f:
        json.dump(test_serializable, f, indent=2)

    ab_qb, ap_qb = _save_qualitative_grid(G, test_loader, mode, use_text, dev, name, run_dir)
    _save_chroma_histogram(ab_qb, ap_qb, name, run_dir)

    print(f"  TEST: {test_serializable}")
    return test_metrics


# ── Summary tables and analysis figures ────────────────────────────────────

def print_summary_tables(R):
    rows = [{
        "experiment": k, "PSNR": f"{v['psnr']:.2f}", "SSIM": f"{v['ssim']:.4f}",
        "LPIPS↓": f"{v['lpips']:.4f}", "ΔE↓": f"{v['delta_e']:.2f}",
        "FID↓ (2048)": f"{v['fid_2048']:.1f}",
        "Chroma↑": f"{v.get('chroma_ratio', 0):.3f}",
    } for k, v in R.items()]
    df = pd.DataFrame(rows)
    print("\nFID: InceptionV3 feature=2048 (standard pool3, NOT reduced).")
    print("LPIPS: AlexNet backbone, reported as eval metric.\n")
    print(df.to_string(index=False))

    print("\n── Text Effect: Baseline → Ours → +Chroma ──")
    keys = [k for k in ["baseline", "ours", "ours_chroma"] if k in R]
    if keys:
        print(pd.DataFrame([{
            "experiment": k, "PSNR": f"{R[k]['psnr']:.2f}",
            "SSIM": f"{R[k]['ssim']:.4f}", "ΔE↓": f"{R[k]['delta_e']:.2f}",
            "Chroma↑": f"{R[k].get('chroma_ratio', 0):.3f}",
        } for k in keys]).to_string(index=False))

    print("\n── Probabilistic Decoding: reg vs MDN ──")
    keys = [k for k in ["ours", "ours_chroma", "ours_mdn"] if k in R]
    if keys:
        print(pd.DataFrame([{
            "experiment": k, "PSNR": f"{R[k]['psnr']:.2f}",
            "ΔE↓": f"{R[k]['delta_e']:.2f}",
            "FID↓": f"{R[k]['fid_2048']:.1f}",
            "Chroma↑": f"{R[k].get('chroma_ratio', 0):.3f}",
        } for k in keys]).to_string(index=False))


def _build_panel(test_paths, test_caps, test_tc, dev, cfg):
    PANEL_CATS = {
        "sky":        ["sky", "cloud", "clouds", "sunset", "sunrise", "horizon"],
        "vegetation": ["grass", "tree", "trees", "forest", "garden", "plant", "leaves", "flower"],
        "snow":       ["snow", "mountain", "ice", "winter", "frozen"],
        "animals":    ["dog", "cat", "horse", "bird", "cow", "sheep", "elephant", "bear", "animal"],
        "urban":      ["street", "building", "city", "road", "traffic", "bus", "car", "bridge"],
    }

    panel_idx = []
    if test_caps:
        for cat, kws in PANEL_CATS.items():
            found = []
            for ci, (_, _, tags) in enumerate(test_caps):
                if any(k in tags.lower() for k in kws) and ci not in panel_idx:
                    found.append(ci)
                if len(found) >= 2:
                    break
            panel_idx.extend(found[:2])
    if len(panel_idx) < 8:
        extra = [i for i in range(min(50, len(test_paths))) if i not in panel_idx]
        panel_idx.extend(extra[:8 - len(panel_idx)])
    panel_idx = panel_idx[:12]
    print(f"Fixed evaluation panel: {len(panel_idx)} images")

    S = cfg["img_size"]
    pL, pAB, pRGB, pTK, pMK = [], [], [], [], []
    for pi in panel_idx:
        img = Image.open(test_paths[pi]).convert("RGB").resize((S, S))
        rgb_np = np.array(img).astype(np.float32) / 255
        lab = skcolor.rgb2lab(rgb_np)
        pL.append(torch.from_numpy(lab[..., 0:1] / 100).permute(2, 0, 1).float())
        pAB.append(torch.from_numpy(lab[..., 1:3] / 128).permute(2, 0, 1).float())
        pRGB.append(torch.from_numpy(rgb_np).permute(2, 0, 1).float())
        e = test_tc.get(pi)
        if e:
            pTK.append(e["tokens"]); pMK.append(e["mask"])
        else:
            pTK.append(torch.zeros(1, 512))
            pMK.append(torch.zeros(1, dtype=torch.long))

    pL  = torch.stack(pL).to(dev)
    pAB = torch.stack(pAB).to(dev)
    pRGB = torch.stack(pRGB).to(dev)

    mT = max(t.size(0) for t in pTK)
    pTP = torch.zeros(len(panel_idx), mT, 512)
    pMP = torch.zeros(len(panel_idx), mT, dtype=torch.long)
    for i, (t, m) in enumerate(zip(pTK, pMK)):
        T = t.size(0); pTP[i, :T] = t; pMP[i, :T] = m
    return panel_idx, pL, pAB, pRGB, pTP.to(dev), pMP.to(dev)


def comparison_figures(R, test_paths, test_caps, test_tc, dev, cfg):
    if not R:
        return
    panel_idx, pL, pAB, pRGB, pTP, pMP = _build_panel(test_paths, test_caps, test_tc, dev, cfg)
    n_p = len(panel_idx)

    core_models = [(k, R[k]["model"], k != "baseline")
                   for k in ["baseline", "ours", "ours_chroma"] if k in R]
    mdn_key = "ours_mdn" if "ours_mdn" in R else None
    ncols = 2 + len(core_models) + (2 if mdn_key else 0)
    col_titles = (["Grayscale", "Ground Truth"]
                  + ["Baseline", "Ours", "Ours + Lc"][:len(core_models)]
                  + (["MDN expected", "MDN sampled"] if mdn_key else []))

    fig, axes = plt.subplots(n_p, ncols, figsize=(3 * ncols, 3 * n_p))
    if n_p == 1:
        axes = axes[np.newaxis, :]
    for i in range(n_p):
        axes[i, 0].imshow(pL[i, 0].cpu().numpy(), cmap="gray"); axes[i, 0].axis("off")
        axes[i, 1].imshow(pRGB[i].permute(1, 2, 0).cpu().numpy()); axes[i, 1].axis("off")
        for j, (mn, M, ut) in enumerate(core_models):
            M.eval()
            with torch.no_grad():
                out = M(pL[i:i+1],
                        text_tokens=pTP[i:i+1] if ut else None,
                        text_mask=pMP[i:i+1] if ut else None,
                        use_text=ut)
                ap = out["ab"]
                rp = lab_to_rgb_np(pL[i:i+1], ap).clamp(0, 1)
            axes[i, 2 + j].imshow(rp[0].permute(1, 2, 0).cpu().numpy())
            axes[i, 2 + j].axis("off")
        if mdn_key:
            Mm = R[mdn_key]["model"]; Mm.eval()
            with torch.no_grad():
                om = Mm(pL[i:i+1], text_tokens=pTP[i:i+1], text_mask=pMP[i:i+1], use_text=True)
                rpe = lab_to_rgb_np(pL[i:i+1], mdn_expected_ab(om["pi"], om["mu_mdn"])).clamp(0, 1)
                rps = lab_to_rgb_np(pL[i:i+1],
                                    mdn_sample_ab(om["pi"], om["mu_mdn"], om["log_sigma"], 1.0)).clamp(0, 1)
            ci = 2 + len(core_models)
            axes[i, ci].imshow(rpe[0].permute(1, 2, 0).cpu().numpy()); axes[i, ci].axis("off")
            axes[i, ci + 1].imshow(rps[0].permute(1, 2, 0).cpu().numpy()); axes[i, ci + 1].axis("off")
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=9)
    plt.suptitle("Main Comparison — Fixed Evaluation Panel", fontsize=14)
    plt.tight_layout()
    out_path = os.path.join(cfg["save_dir"], "main_comparison_grid.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    if "ours" in R and "ours_chroma" in R:
        M_ta = R["ours"]["model"]; M_tc = R["ours_chroma"]["model"]
        M_ta.eval(); M_tc.eval()
        fig_cr, ax_cr = plt.subplots(n_p, 4, figsize=(12, 3 * n_p), squeeze=False)
        for i in range(n_p):
            ax_cr[i, 0].imshow(pL[i, 0].cpu().numpy(), cmap="gray"); ax_cr[i, 0].axis("off")
            ax_cr[i, 1].imshow(pRGB[i].permute(1, 2, 0).cpu().numpy()); ax_cr[i, 1].axis("off")
            with torch.no_grad():
                o1 = M_ta(pL[i:i+1], text_tokens=pTP[i:i+1], text_mask=pMP[i:i+1], use_text=True)
                o2 = M_tc(pL[i:i+1], text_tokens=pTP[i:i+1], text_mask=pMP[i:i+1], use_text=True)
                r1 = lab_to_rgb_np(pL[i:i+1], o1["ab"]).clamp(0, 1)
                r2 = lab_to_rgb_np(pL[i:i+1], o2["ab"]).clamp(0, 1)
            ax_cr[i, 2].imshow(r1[0].permute(1, 2, 0).cpu().numpy()); ax_cr[i, 2].axis("off")
            ax_cr[i, 3].imshow(r2[0].permute(1, 2, 0).cpu().numpy()); ax_cr[i, 3].axis("off")
        for j, t in enumerate(["L", "GT", "Ours", "Ours + Lc"]):
            ax_cr[0, j].set_title(t, fontsize=9)
        plt.suptitle("Chroma Recovery: Effect of Chroma-Aware Loss", fontsize=14)
        plt.tight_layout()
        fig_cr.savefig(os.path.join(cfg["save_dir"], "chroma_recovery.png"),
                       dpi=150, bbox_inches="tight")
        plt.close(fig_cr)
        print("Saved chroma_recovery.png")

    if "baseline" in R and "ours" in R:
        M_nv = R["baseline"]["model"]; M_tx = R["ours"]["model"]
        M_nv.eval(); M_tx.eval()
        fig_cg, ax_cg = plt.subplots(n_p, 4, figsize=(12, 3 * n_p), squeeze=False)
        for i in range(n_p):
            ax_cg[i, 0].imshow(pL[i, 0].cpu().numpy(), cmap="gray"); ax_cg[i, 0].axis("off")
            ax_cg[i, 1].imshow(pRGB[i].permute(1, 2, 0).cpu().numpy()); ax_cg[i, 1].axis("off")
            with torch.no_grad():
                onv = M_nv(pL[i:i+1], use_text=False)
                otx = M_tx(pL[i:i+1], text_tokens=pTP[i:i+1], text_mask=pMP[i:i+1], use_text=True)
                rnv = lab_to_rgb_np(pL[i:i+1], onv["ab"]).clamp(0, 1)
                rtx = lab_to_rgb_np(pL[i:i+1], otx["ab"]).clamp(0, 1)
            ax_cg[i, 2].imshow(rnv[0].permute(1, 2, 0).cpu().numpy()); ax_cg[i, 2].axis("off")
            ax_cg[i, 3].imshow(rtx[0].permute(1, 2, 0).cpu().numpy()); ax_cg[i, 3].axis("off")
        for j, t in enumerate(["L", "GT", "Baseline", "Ours"]):
            ax_cg[0, j].set_title(t, fontsize=9)
        plt.suptitle("Caption Guidance: Baseline vs Text-Conditioned", fontsize=14)
        plt.tight_layout()
        fig_cg.savefig(os.path.join(cfg["save_dir"], "caption_guidance.png"),
                       dpi=150, bbox_inches="tight")
        plt.close(fig_cg)
        print("Saved caption_guidance.png")


# ── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train colorization ablations.")
    p.add_argument("--n-images", type=int, default=-1,
                   help="Limit dataset size (-1 = use all available).")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override CONFIG['epochs'].")
    p.add_argument("--batch-size", type=int, default=None,
                   help="Override CONFIG['batch_size'].")
    p.add_argument("--experiments", type=str, default="baseline,ours,ours_chroma,ours_mdn",
                   help="Comma-separated subset of experiments to run.")
    p.add_argument("--no-mount-drive", action="store_true",
                   help="Skip Google Drive mount on Colab.")
    p.add_argument("--skip-comparison-figures", action="store_true",
                   help="Skip the panel/chroma/caption comparison figures.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = default_config(n_images=args.n_images, mount_drive=not args.no_mount_drive)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size

    seed_all(cfg["seed"])
    print(f"Device: {cfg['device']} | n_images: {cfg['n_images']} | cache_text: {cfg['cache_text']}")
    print(f"Images : {cfg['images_dir']}")
    print(f"Runs   : {cfg['save_dir']}")

    # 1. Dataset
    ensure_images(cfg["images_dir"])
    image_paths = list_images(cfg["images_dir"])
    assert image_paths, f"No images found in {cfg['images_dir']}"
    if cfg["n_images"] > 0 and cfg["n_images"] < len(image_paths):
        image_paths = image_paths[-cfg["n_images"]:]
        print(f"Using last {len(image_paths)} of available images")

    train_paths, val_paths, test_paths = make_or_load_split(image_paths, cfg)
    print(f"Train:{len(train_paths)} Val:{len(val_paths)} Test:{len(test_paths)}")

    # 2. Loaders
    datasets, plain_loaders = make_plain_loaders(train_paths, val_paths, test_paths, cfg)
    train_loader, val_loader, test_loader = plain_loaders

    # 3. Text caches
    train_tc, train_caps = build_text_cache(train_paths, "train", cfg["device"], cfg)
    val_tc,   val_caps   = build_text_cache(val_paths,   "val",   cfg["device"], cfg)
    test_tc,  test_caps  = build_text_cache(test_paths,  "test",  cfg["device"], cfg)
    print(f"Text: tr={len(train_tc)} vl={len(val_tc)} te={len(test_tc)}")

    tr_tl, vl_tl, te_tl = make_text_loaders(datasets, (train_tc, val_tc, test_tc), cfg)
    print("Text loaders ready")

    # 4. Run ablations
    requested = {x.strip() for x in args.experiments.split(",") if x.strip()}
    R = {}
    spec = {
        "baseline":    dict(mode="reg", use_text=False, text_mode="none",
                            loaders=(train_loader, val_loader, test_loader),
                            lambda_chroma=0.0),
        "ours":        dict(mode="reg", use_text=True,  text_mode="token_attn",
                            loaders=(tr_tl, vl_tl, te_tl), lambda_chroma=0.0),
        "ours_chroma": dict(mode="reg", use_text=True,  text_mode="token_attn",
                            loaders=(tr_tl, vl_tl, te_tl),
                            lambda_chroma=cfg["lambda_chroma"]),
        "ours_mdn":    dict(mode="mdn", use_text=True,  text_mode="multiscale_film",
                            loaders=(tr_tl, vl_tl, te_tl), lambda_chroma=0.0),
    }
    for name in ["baseline", "ours", "ours_chroma", "ours_mdn"]:
        if name not in requested:
            continue
        s = spec[name]
        R[name] = run_experiment(
            name, s["mode"], s["use_text"], s["text_mode"],
            s["loaders"][0], s["loaders"][1], s["loaders"][2],
            cfg["device"], cfg, lambda_chroma=s["lambda_chroma"],
        )

    # 5. Tables and figures
    if R:
        print_summary_tables(R)
        if not args.skip_comparison_figures:
            comparison_figures(R, test_paths, test_caps, test_tc, cfg["device"], cfg)

    print("\nDone.")


if __name__ == "__main__":
    main()
