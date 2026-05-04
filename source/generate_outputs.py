"""Generate visual outputs from trained checkpoints.

Equivalent to `test_outputs.ipynb`. Reads the per-experiment run directories
written by `train.py`, then writes:

  * tab_1_main_results.{tex,csv}       -- ablation table
  * tab_2_per_category.{tex,csv}       -- per-category dE
  * tab_3_mdn_decoding.{tex,csv}       -- MDN decoding strategies
  * fig_1_architecture.pdf             -- schematic
  * fig_2_qualitative_grid.pdf         -- per-category visual grid
  * fig_3_chroma_recovery.pdf          -- chroma collapse vs +Lc
  * fig_4_mdn_diversity.pdf            -- expected vs sampled MDN outputs
  * fig_5_1_training_deltae.pdf        -- val dE training curves
  * fig_5_2_training_psnr.pdf          -- val PSNR training curves
  * fig_6_caption_examples.pdf         -- BLIP caption + tag examples

Run:    python -m source.generate_outputs
        python -m source.generate_outputs --eval-max-batches 100
"""

import argparse
import json
import os

import lpips
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage import color as skcolor
from torchmetrics.image import (
    PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure,
)
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm.auto import tqdm

from .colorlab import (
    ciede2000_batch, lab_to_rgb_np,
    mdn_best_of_n, mdn_expected_ab, mdn_map_ab, mdn_sample_ab,
    to_uint8,
)
from .config import default_config, seed_all
from .data import (
    list_images, make_or_load_split,
    make_plain_loaders, make_text_loaders, unpack_batch,
)
from .dataset_setup import ensure_images
from .models import UNetColorizer, UNetTextColorizer
from .text_cache import build_text_cache


EXPERIMENTS = ["baseline", "ours", "ours_chroma", "ours_mdn"]
MODEL_LABELS = {
    "baseline":    "Baseline",
    "ours":        "Ours",
    "ours_chroma": "Ours+$\\mathcal{L}_c$",
    "ours_mdn":    "Ours+MDN",
}
CATEGORIES = {
    "sky":        ["sky", "cloud", "clouds", "sunset", "sunrise", "horizon", "blue"],
    "vegetation": ["grass", "tree", "trees", "forest", "garden", "plant", "leaves", "flower", "green"],
    "snow":       ["snow", "mountain", "ice", "winter", "frozen"],
    "animals":    ["dog", "cat", "horse", "bird", "cow", "sheep", "elephant", "bear", "animal", "fish"],
    "urban":      ["street", "building", "city", "road", "traffic", "bus", "car", "train", "bridge"],
}


def setup_visual_plotting():
    """Adopt a compact serif matplotlib style."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'axes.linewidth': 0.5,
        'grid.linewidth': 0.3,
        'lines.linewidth': 1.0,
    })


# ── Loading ────────────────────────────────────────────────────────────────

def load_experiments(results_dir: str, dev: str, K: int):
    """Instantiate the right model class for each saved experiment and load weights."""
    R = {}
    for name in EXPERIMENTS:
        exp_dir = os.path.join(results_dir, name)
        ckpt_path = os.path.join(exp_dir, "best_val_checkpoint.pt")
        cfg_path  = os.path.join(exp_dir, "config.json")
        if not os.path.exists(ckpt_path):
            print(f"WARNING: No checkpoint for {name} at {ckpt_path}")
            continue
        if not os.path.exists(cfg_path):
            print(f"WARNING: Missing config.json for {name}; skipping")
            continue

        with open(cfg_path) as f:
            exp_cfg = json.load(f)
        mode = exp_cfg.get("mode", "reg")
        text_mode = exp_cfg.get("text_mode", "none")
        use_text = exp_cfg.get("use_text", False) and text_mode != "none"

        if use_text:
            model = UNetTextColorizer(text_in_dim=512, K=K, mode=mode, text_mode=text_mode).to(dev)
        else:
            model = UNetColorizer(K=K, mode=mode).to(dev)
        state = torch.load(ckpt_path, map_location=dev, weights_only=False)
        model.load_state_dict(state)
        model.eval()

        R[name] = {
            "model": model, "mode": mode, "use_text": use_text,
            "text_mode": text_mode, "exp_dir": exp_dir,
        }
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Loaded {name}: {ckpt_path} ({n_params:,} params, mode={mode}, text={text_mode})")

    print(f"\nLoaded {len(R)} models: {list(R.keys())}")
    return R


def fill_metrics(R, te_text_loader, test_loader, dev, cfg, lpips_fn):
    """Reuse on-disk test_metrics.json when available; recompute otherwise."""
    for name, info in R.items():
        saved_metrics_path = os.path.join(info["exp_dir"], "test_metrics.json")
        if os.path.exists(saved_metrics_path):
            with open(saved_metrics_path) as f:
                saved = json.load(f)
            info.update({
                "psnr":         saved["psnr"],
                "ssim":         saved["ssim"],
                "fid_2048":     saved["fid_2048"],
                "lpips":        saved["lpips"],
                "delta_e":      saved["delta_e"],
                "chroma_ratio": saved["chroma_ratio"],
            })
            print(f"  {name} (from saved): PSNR={info['psnr']:.2f} SSIM={info['ssim']:.4f} "
                  f"LPIPS={info['lpips']:.4f} dE={info['delta_e']:.2f} "
                  f"FID={info['fid_2048']:.1f} Chroma={info['chroma_ratio']:.3f}")
            continue

        M = info["model"]; M.eval()
        ut = info["use_text"]; mode = info["mode"]
        loader = te_text_loader if ut else test_loader

        psnr = PeakSignalNoiseRatio(data_range=1.0).to(dev)
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(dev)
        fid  = FrechetInceptionDistance(feature=2048).to(dev)
        lps, des, c_prs, c_gts = [], [], [], []
        with torch.no_grad():
            for bi, b in enumerate(tqdm(loader, desc=f"Eval {name}")):
                if cfg.get("eval_max_batches") and bi >= cfg["eval_max_batches"]:
                    break
                L, ab, rgb, tk, mk = unpack_batch(b, ut, dev)
                out = M(L, text_tokens=tk, text_mask=mk, use_text=ut)
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
        info.update({
            "psnr":         float(psnr.compute().item()),
            "ssim":         float(ssim.compute().item()),
            "fid_2048":     float(fid.compute().item()),
            "lpips":        float(np.mean(lps)) if lps else 0.0,
            "delta_e":      float(np.mean(des)) if des else 0.0,
            "chroma_ratio": float(cr),
        })
        print(f"  {name}: PSNR={info['psnr']:.2f} SSIM={info['ssim']:.4f} "
              f"LPIPS={info['lpips']:.4f} dE={info['delta_e']:.2f} "
              f"FID={info['fid_2048']:.1f} Chroma={info['chroma_ratio']:.3f}")
    print("\nAll models evaluated.")


# ── Tables ──────────────────────────────────────────────────────────────────

def write_table_main(R, out_dir):
    labels = [
        ("baseline",    "Baseline",                   "\\ding{55}", "\\ding{55}", "Reg"),
        ("ours",        "Ours",                       "\\ding{51}", "\\ding{55}", "Reg"),
        ("ours_chroma", "Ours + $\\mathcal{L}_c$",    "\\ding{51}", "\\ding{51}", "Reg"),
        ("ours_mdn",    "Ours + MDN",                 "\\ding{51}", "\\ding{55}", "MDN"),
    ]
    rows = []
    for key, label, text_mark, chroma_mark, head in labels:
        if key not in R:
            continue
        v = R[key]
        rows.append({
            "Model": label, "Text": text_mark, "Chroma": chroma_mark, "Head": head,
            "PSNR": v["psnr"], "SSIM": v["ssim"], "LPIPS": v["lpips"],
            "dE": v["delta_e"], "FID": v["fid_2048"], "Chroma_ratio": v["chroma_ratio"],
        })
    df = pd.DataFrame(rows)
    assert len(df) > 0, "No models loaded -- check that results folder has checkpoint subdirectories"

    delta_row = None
    if "baseline" in R and "ours" in R:
        vb, vo = R["baseline"], R["ours"]
        delta_row = {
            "Model": "Delta (Ours - Baseline)", "Text": "", "Chroma": "", "Head": "",
            "PSNR":         vo["psnr"]         - vb["psnr"],
            "SSIM":         vo["ssim"]         - vb["ssim"],
            "LPIPS":        vo["lpips"]        - vb["lpips"],
            "dE":           vo["delta_e"]      - vb["delta_e"],
            "FID":          vo["fid_2048"]     - vb["fid_2048"],
            "Chroma_ratio": vo["chroma_ratio"] - vb["chroma_ratio"],
        }
        df_with_delta = pd.concat([df, pd.DataFrame([delta_row])], ignore_index=True)
    else:
        df_with_delta = df

    csv_path = os.path.join(out_dir, "tab_1_main_results.csv")
    df_with_delta.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    best_psnr  = df["PSNR"].max();  best_ssim = df["SSIM"].max()
    best_lpips = df["LPIPS"].min(); best_de   = df["dE"].min()
    best_fid   = df["FID"].min()

    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Ablation study on COCO 2017 test set. Best values in \textbf{bold}.}",
        r"\label{tab:main}",
        r"\small",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"Model & Text & $\mathcal{L}_c$ & Head & PSNR$\uparrow$ & SSIM$\uparrow$ & LPIPS$\downarrow$ & $\Delta E$$\downarrow$ & FID$\downarrow$ \\",
        r"\midrule",
    ]

    def bold_if(v, best, fmt):
        s = f"{v:{fmt}}"
        return f"\\textbf{{{s}}}" if abs(v - best) < 1e-6 else s

    for _, row in df.iterrows():
        line = (f"{row['Model']} & {row['Text']} & {row['Chroma']} & {row['Head']} & "
                f"{bold_if(row['PSNR'],  best_psnr,  '.2f')} & "
                f"{bold_if(row['SSIM'],  best_ssim,  '.4f')} & "
                f"{bold_if(row['LPIPS'], best_lpips, '.4f')} & "
                f"{bold_if(row['dE'],    best_de,    '.2f')} & "
                f"{bold_if(row['FID'],   best_fid,   '.1f')} \\\\")
        tex_lines.append(line)

    if delta_row:
        d = delta_row
        tex_lines.append(r"\midrule")
        tex_lines.append(
            f"$\\Delta$ & & & & "
            f"{d['PSNR']:+.2f} & {d['SSIM']:+.4f} & {d['LPIPS']:+.4f} & "
            f"{d['dE']:+.2f} & {d['FID']:+.1f} \\\\"
        )

    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_str = "\n".join(tex_lines)
    tex_path = os.path.join(out_dir, "tab_1_main_results.tex")
    with open(tex_path, "w") as f:
        f.write(tex_str)
    print(f"Saved: {tex_path}")
    print("\n" + tex_str)
    print("\n" + df_with_delta.to_string(index=False))


def _categorize(test_caps, n_test_paths):
    """Bucket test images by tag-derived category."""
    cat_idx = {c: [] for c in CATEGORIES}
    if not test_caps:
        return cat_idx
    for ci, (_, _, tags) in enumerate(test_caps[:n_test_paths]):
        for cat, kws in CATEGORIES.items():
            if any(k in tags.lower() for k in kws):
                cat_idx[cat].append(ci)
                break
    return cat_idx


def write_table_per_category(R, test_paths, test_caps, test_tc, dev, cfg, out_dir):
    cat_idx = _categorize(test_caps, len(test_paths))
    models_eval = [(k, R[k]["model"], R[k]["use_text"]) for k in EXPERIMENTS if k in R]
    S = cfg["img_size"]

    cat_results = {}
    for cat in CATEGORIES:
        idxs = cat_idx[cat][:8]
        if len(idxs) < 2:
            continue
        cat_results[cat] = {"n": len(idxs)}
        for mn, M, ut in models_eval:
            M.eval(); des = []
            for ti in idxs:
                img = Image.open(test_paths[ti]).convert("RGB").resize((S, S))
                rgb_np = np.array(img).astype(np.float32) / 255
                lab = skcolor.rgb2lab(rgb_np)
                Lt  = torch.from_numpy(lab[..., 0:1] / 100).permute(2, 0, 1).float().unsqueeze(0).to(dev)
                abt = torch.from_numpy(lab[..., 1:3] / 128).permute(2, 0, 1).float().unsqueeze(0).to(dev)
                e = test_tc.get(ti)
                tk_i = e["tokens"].unsqueeze(0).to(dev) if e else None
                mk_i = e["mask"].unsqueeze(0).to(dev) if e else None
                with torch.no_grad():
                    out = M(Lt,
                            text_tokens=tk_i if ut else None,
                            text_mask=mk_i if ut else None, use_text=ut)
                    ap = out["ab"] if M.mode == "reg" else mdn_expected_ab(out["pi"], out["mu_mdn"])
                    des.append(ciede2000_batch(Lt, abt, ap))
            cat_results[cat][mn] = float(np.mean(des))

    if not cat_results:
        print("Per-category table skipped -- not enough categorized images")
        return

    cat_rows = []
    for cat, cr in cat_results.items():
        base_de = cr.get("baseline", 0)
        des_by_model = {mn: cr.get(mn, 999) for mn, _, _ in models_eval}
        best_de = min(des_by_model.values())
        improvement = (base_de - best_de) / base_de * 100 if base_de and best_de < base_de else 0.0
        cat_rows.append({
            "Category": cat.capitalize(), "N": cr["n"],
            **{mn: cr.get(mn, 0) for mn, _, _ in models_eval},
            "best_delta_pct": improvement,
        })
    cat_rows.sort(key=lambda x: -x["best_delta_pct"])
    df_cat = pd.DataFrame(cat_rows)

    csv_path = os.path.join(out_dir, "tab_2_per_category.csv")
    df_cat.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    model_keys = [mn for mn, _, _ in models_eval]
    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Per-category $\Delta E$ (CIEDE2000, lower is better). Best in \textbf{bold}.}",
        r"\label{tab:category}",
        r"\small",
        r"\begin{tabular}{" + ("l" + "c" * (len(model_keys) + 2)) + "}",
        r"\toprule",
        "Category & N & " + " & ".join([MODEL_LABELS.get(mn, mn) for mn in model_keys]) + r" & Best $\Delta$ \\",
        r"\midrule",
    ]
    for _, row in df_cat.iterrows():
        vals = [row[mn] for mn in model_keys]
        best_v = min(vals)
        cells = [(f"\\textbf{{{v:.2f}}}" if abs(v - best_v) < 0.005 else f"{v:.2f}") for v in vals]
        delta_str = f"$-${row['best_delta_pct']:.1f}\\%" if row["best_delta_pct"] > 0 else "---"
        tex_lines.append(f"{row['Category']} & {row['N']} & " + " & ".join(cells) + f" & {delta_str} \\\\")
    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    tex_str = "\n".join(tex_lines)
    tex_path = os.path.join(out_dir, "tab_2_per_category.tex")
    with open(tex_path, "w") as f:
        f.write(tex_str)
    print(f"Saved: {tex_path}")
    print("\n" + tex_str)
    print("\n" + df_cat.to_string(index=False))
    return cat_idx


def write_table_mdn_decoding(R, te_text_loader, dev, cfg, out_dir, lpips_fn):
    if "ours_mdn" not in R:
        print("Skipping Table 3 -- need ours_mdn model")
        return
    mdn_M = R["ours_mdn"]["model"]; mdn_M.eval()

    all_L, all_ab, all_rgb = [], [], []
    all_out = {"pi": [], "mu_mdn": [], "log_sigma": []}
    with torch.no_grad():
        for bi, b in enumerate(tqdm(te_text_loader, desc="MDN decode eval")):
            if bi >= cfg["eval_max_batches"]:
                break
            L, ab, rgb, tk, mk = unpack_batch(b, True, dev)
            out = mdn_M(L, text_tokens=tk, text_mask=mk, use_text=True)
            all_L.append(L); all_ab.append(ab); all_rgb.append(rgb)
            for k in all_out:
                all_out[k].append(out[k])
    all_L = torch.cat(all_L); all_ab = torch.cat(all_ab); all_rgb = torch.cat(all_rgb)
    for k in all_out:
        all_out[k] = torch.cat(all_out[k])

    with torch.no_grad():
        decode_modes = {
            "Expected ($\\mathbb{E}[ab]$)": mdn_expected_ab(all_out["pi"], all_out["mu_mdn"]),
            "MAP ($\\arg\\max \\pi$)":      mdn_map_ab(all_out["pi"], all_out["mu_mdn"]),
            "Best-of-5":                    mdn_best_of_n(all_out["pi"], all_out["mu_mdn"],
                                                          all_out["log_sigma"], all_L, all_ab, 5),
            "Sample ($T$=1.0)":             mdn_sample_ab(all_out["pi"], all_out["mu_mdn"],
                                                          all_out["log_sigma"], 1.0),
        }

    BS = 32
    decode_results = []
    for dname, dab in decode_modes.items():
        psnr = PeakSignalNoiseRatio(data_range=1.0).to(dev)
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(dev)
        lps, des = [], []
        for si in range(0, len(all_L), BS):
            Li = all_L[si:si + BS]
            abi = all_ab[si:si + BS]
            rgbi = all_rgb[si:si + BS]
            dabi = dab[si:si + BS]
            rp = lab_to_rgb_np(Li, dabi).clamp(0, 1)
            if torch.isnan(rp).any():
                continue
            psnr.update(rp, rgbi); ssim.update(rp, rgbi)
            lps.append(lpips_fn(rp * 2 - 1, rgbi * 2 - 1).mean().item())
            des.append(ciede2000_batch(Li, abi, dabi))
        c_pr = torch.sqrt(dab[:, 0:1] ** 2 + dab[:, 1:2] ** 2 + 1e-8).mean().item()
        c_gt = torch.sqrt(all_ab[:, 0:1] ** 2 + all_ab[:, 1:2] ** 2 + 1e-8).mean().item()
        decode_results.append({
            "Decoding": dname,
            "PSNR":   float(psnr.compute().item()),
            "SSIM":   float(ssim.compute().item()),
            "LPIPS":  float(np.mean(lps)),
            "dE":     float(np.mean(des)),
            "Chroma": c_pr / max(c_gt, 1e-6),
        })

    df_dec = pd.DataFrame(decode_results)
    csv_path = os.path.join(out_dir, "tab_3_mdn_decoding.csv")
    df_dec.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    best_p = max(r["PSNR"]   for r in decode_results)
    best_s = max(r["SSIM"]   for r in decode_results)
    best_l = min(r["LPIPS"]  for r in decode_results)
    best_d = min(r["dE"]     for r in decode_results)
    best_c = max(r["Chroma"] for r in decode_results)

    def bold_if(v, best, fmt):
        s = f"{v:{fmt}}"
        return f"\\textbf{{{s}}}" if abs(v - best) < 1e-4 else s

    tex_lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{MDN decoding strategies on the test set.}",
        r"\label{tab:mdn}",
        r"\small",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Decoding & PSNR$\uparrow$ & SSIM$\uparrow$ & LPIPS$\downarrow$ & $\Delta E$$\downarrow$ & Chroma$\uparrow$ \\",
        r"\midrule",
    ]
    for r in decode_results:
        tex_lines.append(
            f"{r['Decoding']} & {bold_if(r['PSNR'], best_p, '.2f')} & "
            f"{bold_if(r['SSIM'], best_s, '.4f')} & {bold_if(r['LPIPS'], best_l, '.4f')} & "
            f"{bold_if(r['dE'], best_d, '.2f')} & {bold_if(r['Chroma'], best_c, '.3f')} \\\\"
        )
    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    tex_str = "\n".join(tex_lines)
    tex_path = os.path.join(out_dir, "tab_3_mdn_decoding.tex")
    with open(tex_path, "w") as f:
        f.write(tex_str)
    print(f"Saved: {tex_path}")
    print("\n" + tex_str)
    print("\n" + df_dec.to_string(index=False))


# ── Figures ────────────────────────────────────────────────────────────────

def fig_qualitative_grid(R, test_paths, test_tc, cat_idx, dev, cfg, out_dir):
    if not R or "baseline" not in R:
        print("Skipping Figure 2 -- baseline missing")
        return
    S = cfg["img_size"]
    PANEL_CATS = {
        "sky":        ["sky", "cloud", "clouds", "sunset", "sunrise", "horizon"],
        "vegetation": ["grass", "tree", "trees", "forest", "garden", "plant", "leaves", "flower"],
        "snow":       ["snow", "mountain", "ice", "winter", "frozen"],
        "animals":    ["dog", "cat", "horse", "bird", "cow", "sheep", "elephant", "bear", "animal"],
        "urban":      ["street", "building", "city", "road", "traffic", "bus", "car", "train", "bridge"],
    }

    IMGS_PER_CAT = 2
    panel_selection = {}
    base_M = R["baseline"]["model"]; base_M.eval()
    for cat, _ in PANEL_CATS.items():
        cands = cat_idx.get(cat, [])[:12]
        if not cands:
            continue
        scored = []
        for ci in cands:
            img = Image.open(test_paths[ci]).convert("RGB").resize((S, S))
            rgb_np = np.array(img).astype(np.float32) / 255
            lab = skcolor.rgb2lab(rgb_np)
            Lt  = torch.from_numpy(lab[..., 0:1] / 100).permute(2, 0, 1).float().unsqueeze(0).to(dev)
            abt = torch.from_numpy(lab[..., 1:3] / 128).permute(2, 0, 1).float().unsqueeze(0).to(dev)
            with torch.no_grad():
                out = base_M(Lt, use_text=False)
                de = ciede2000_batch(Lt, abt, out["ab"])
            scored.append((de, ci))
        scored.sort(key=lambda x: -x[0])  # highest baseline dE first
        panel_selection[cat] = [ci for _, ci in scored[:IMGS_PER_CAT]]

    cat_order = ["sky", "vegetation", "snow", "animals", "urban"]
    panel_indices, panel_cat_labels = [], []
    for c in cat_order:
        if c not in panel_selection:
            continue
        for i, ci in enumerate(panel_selection[c]):
            panel_indices.append(ci)
            panel_cat_labels.append(c.capitalize() if i == 0 else "")
    if not panel_indices:
        print("Skipping Figure 2 -- empty panel selection")
        return

    n_rows = len(panel_indices)
    model_keys = [k for k in EXPERIMENTS if k in R]
    col_labels = ["Input", "GT"] + ["Baseline", "Ours", "Ours + $\\mathcal{L}_c$", "Ours + MDN"][:len(model_keys)]
    n_cols = 2 + len(model_keys)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.0 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for ri, pi in enumerate(panel_indices):
        img = Image.open(test_paths[pi]).convert("RGB").resize((S, S))
        rgb_np = np.array(img).astype(np.float32) / 255
        lab = skcolor.rgb2lab(rgb_np)
        Lt  = torch.from_numpy(lab[..., 0:1] / 100).permute(2, 0, 1).float().unsqueeze(0).to(dev)
        e = test_tc.get(pi)
        tk_i = e["tokens"].unsqueeze(0).to(dev) if e else None
        mk_i = e["mask"].unsqueeze(0).to(dev) if e else None

        cat_name = panel_cat_labels[ri]
        if cat_name:
            axes[ri, 0].set_ylabel(cat_name, fontsize=8, rotation=90, labelpad=10)

        axes[ri, 0].imshow(Lt.squeeze().cpu().numpy(), cmap="gray"); axes[ri, 0].axis("off")
        axes[ri, 1].imshow(rgb_np); axes[ri, 1].axis("off")

        for mi, mk in enumerate(model_keys):
            M = R[mk]["model"]; M.eval()
            ut = R[mk]["use_text"]; mode = R[mk]["mode"]
            with torch.no_grad():
                out = M(Lt,
                        text_tokens=tk_i if ut else None,
                        text_mask=mk_i if ut else None, use_text=ut)
                ap = out["ab"] if mode == "reg" else mdn_expected_ab(out["pi"], out["mu_mdn"])
                rp = lab_to_rgb_np(Lt, ap).clamp(0, 1)
            axes[ri, 2 + mi].imshow(rp[0].permute(1, 2, 0).cpu().numpy())
            axes[ri, 2 + mi].axis("off")

    for j, t in enumerate(col_labels):
        axes[0, j].set_title(t, fontsize=8)
    plt.subplots_adjust(wspace=0.02, hspace=0.08)
    pdf_path = os.path.join(out_dir, "fig_2_qualitative_grid.pdf")
    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved: {pdf_path}")


def fig_chroma_recovery(R, te_text_loader, test_tc, n_test_paths, dev, cfg, out_dir):
    if not ("baseline" in R and "ours_chroma" in R):
        print("Skipping Figure 3 -- need baseline and ours_chroma models")
        return
    M_base = R["baseline"]["model"]; M_base.eval()
    M_ours = R.get("ours", {}).get("model")
    if M_ours: M_ours.eval()
    M_chroma = R["ours_chroma"]["model"]; M_chroma.eval()

    collapse_scores = []
    with torch.no_grad():
        for bi, b in enumerate(te_text_loader):
            if bi >= 15:
                break
            L, ab, rgb, idx, tk, mk = b
            L, ab, rgb = L.to(dev), ab.to(dev), rgb.to(dev)
            out_b = M_base(L, use_text=False)
            ap_b = out_b["ab"]
            for ii in range(L.size(0)):
                c_p = torch.sqrt(ap_b[ii, 0] ** 2 + ap_b[ii, 1] ** 2 + 1e-8).mean().item()
                c_g = torch.sqrt(ab[ii, 0] ** 2 + ab[ii, 1] ** 2 + 1e-8).mean().item()
                if c_g > 0.05:
                    collapse_scores.append((c_p / max(c_g, 1e-6),
                                            bi * cfg["batch_size"] + ii,
                                            L[ii:ii + 1], ab[ii:ii + 1], rgb[ii:ii + 1]))
    collapse_scores.sort(key=lambda x: x[0])
    picks = collapse_scores[:2]
    if not picks:
        print("Skipping Figure 3 -- no chroma collapse candidates")
        return

    cols = ["Ground Truth", "Baseline"]
    if M_ours: cols.append("Ours")
    cols.append("Ours + $\\mathcal{L}_c$")
    fig, axes = plt.subplots(len(picks), len(cols), figsize=(2.5 * len(cols), 2.5 * len(picks)))
    if len(picks) == 1:
        axes = axes[np.newaxis, :]

    for ri, (_, idx_val, Li, abi, rgbi) in enumerate(picks):
        e = test_tc.get(idx_val % n_test_paths)
        tki = e["tokens"].unsqueeze(0).to(dev) if e else None
        mki = e["mask"].unsqueeze(0).to(dev) if e else None

        axes[ri, 0].imshow(rgbi[0].permute(1, 2, 0).cpu().numpy()); axes[ri, 0].axis("off")
        c_gt = torch.sqrt(abi[0, 0] ** 2 + abi[0, 1] ** 2 + 1e-8).mean().item()

        with torch.no_grad():
            out_b = M_base(Li, use_text=False)
            rp_b = lab_to_rgb_np(Li, out_b["ab"]).clamp(0, 1)
            c_b = torch.sqrt(out_b["ab"][0, 0] ** 2 + out_b["ab"][0, 1] ** 2 + 1e-8).mean().item()
            axes[ri, 1].imshow(rp_b[0].permute(1, 2, 0).cpu().numpy()); axes[ri, 1].axis("off")
            axes[ri, 1].text(0.5, -0.02, f"C={c_b/max(c_gt,1e-6):.2f}",
                             transform=axes[ri, 1].transAxes,
                             ha='center', va='top', fontsize=6, color='red')
            ci = 2
            if M_ours:
                out_o = M_ours(Li, text_tokens=tki, text_mask=mki, use_text=True)
                rp_o = lab_to_rgb_np(Li, out_o["ab"]).clamp(0, 1)
                c_o = torch.sqrt(out_o["ab"][0, 0] ** 2 + out_o["ab"][0, 1] ** 2 + 1e-8).mean().item()
                axes[ri, ci].imshow(rp_o[0].permute(1, 2, 0).cpu().numpy()); axes[ri, ci].axis("off")
                axes[ri, ci].text(0.5, -0.02, f"C={c_o/max(c_gt,1e-6):.2f}",
                                  transform=axes[ri, ci].transAxes,
                                  ha='center', va='top', fontsize=6, color='orange')
                ci += 1
            out_c = M_chroma(Li, text_tokens=tki, text_mask=mki, use_text=True)
            rp_c = lab_to_rgb_np(Li, out_c["ab"]).clamp(0, 1)
            c_c = torch.sqrt(out_c["ab"][0, 0] ** 2 + out_c["ab"][0, 1] ** 2 + 1e-8).mean().item()
            axes[ri, ci].imshow(rp_c[0].permute(1, 2, 0).cpu().numpy()); axes[ri, ci].axis("off")
            axes[ri, ci].text(0.5, -0.02, f"C={c_c/max(c_gt,1e-6):.2f}",
                              transform=axes[ri, ci].transAxes,
                              ha='center', va='top', fontsize=6, color='green')

    for j, t in enumerate(cols):
        axes[0, j].set_title(t, fontsize=8)
    plt.subplots_adjust(wspace=0.02, hspace=0.08)
    pdf_path = os.path.join(out_dir, "fig_3_chroma_recovery.pdf")
    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved: {pdf_path}")


def fig_mdn_diversity(R, te_text_loader, dev, out_dir):
    if "ours_mdn" not in R:
        print("Skipping Figure 4 -- need ours_mdn model")
        return
    mdn_M = R["ours_mdn"]["model"]; mdn_M.eval()

    best_var, best_data = -1.0, None
    with torch.no_grad():
        for bi, b in enumerate(te_text_loader):
            if bi >= 10:
                break
            L, ab, rgb, idx, tk, mk = b
            L, ab, rgb = L.to(dev), ab.to(dev), rgb.to(dev)
            tk, mk = tk.to(dev), mk.to(dev)
            for ii in range(L.size(0)):
                ab_var = ab[ii].var().item()
                if ab_var > best_var:
                    best_var = ab_var
                    best_data = (L[ii:ii + 1], ab[ii:ii + 1], rgb[ii:ii + 1],
                                 tk[ii:ii + 1], mk[ii:ii + 1])
    if best_data is None:
        print("Skipping Figure 4 -- no test data found")
        return

    Li, abi, rgbi, tki, mki = best_data
    with torch.no_grad():
        out = mdn_M(Li, text_tokens=tki, text_mask=mki, use_text=True)
        ab_exp  = mdn_expected_ab(out["pi"], out["mu_mdn"])
        samples = {t: mdn_sample_ab(out["pi"], out["mu_mdn"], out["log_sigma"], t)
                   for t in [0.5, 1.0, 1.5]}

    cols = ["Grayscale", "GT", "Expected", "T=0.5", "T=1.0", "T=1.5"]
    images = [
        Li[0, 0].cpu().numpy(),
        rgbi[0].permute(1, 2, 0).cpu().numpy(),
        lab_to_rgb_np(Li, ab_exp).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy(),
    ] + [
        lab_to_rgb_np(Li, samples[t]).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
        for t in [0.5, 1.0, 1.5]
    ]

    fig, axes = plt.subplots(1, len(cols), figsize=(2.2 * len(cols), 2.2))
    for j, (ax, img, title) in enumerate(zip(axes, images, cols)):
        if j == 0:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(np.clip(img, 0, 1))
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    plt.subplots_adjust(wspace=0.02)
    pdf_path = os.path.join(out_dir, "fig_4_mdn_diversity.pdf")
    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved: {pdf_path}")


def fig_training_curves(results_dir, out_dir):
    """Read each experiment's metrics_history.csv and plot val dE / val PSNR."""
    hist_map = {
        "Baseline":                 "baseline",
        "Ours":                     "ours",
        "Ours + $\\mathcal{L}_c$":  "ours_chroma",
        "Ours + MDN":               "ours_mdn",
    }
    colors_map = {
        "Baseline":                "#1f77b4",
        "Ours":                    "#ff7f0e",
        "Ours + $\\mathcal{L}_c$": "#2ca02c",
        "Ours + MDN":              "#d62728",
    }

    hist_data = {}
    for label, dirname in hist_map.items():
        csv_path = os.path.join(results_dir, dirname, "metrics_history.csv")
        if not os.path.exists(csv_path):
            print(f"  No history CSV for {dirname}")
            continue
        df_h = pd.read_csv(csv_path)
        if "epoch" not in df_h.columns:
            continue
        hist_data[label] = df_h

    if not hist_data:
        print("No training history CSVs found -- skipping Figure 5")
        return

    fig1, ax1 = plt.subplots(1, 1, figsize=(4.5, 2.8))
    for label, df_h in hist_data.items():
        c = colors_map.get(label, None)
        if "val_deltae" in df_h.columns:
            ax1.plot(df_h["epoch"], df_h["val_deltae"], label=label, color=c)
            ax1.scatter(df_h["epoch"].iloc[-1], df_h["val_deltae"].iloc[-1],
                        marker='o', s=25, color=c, zorder=5)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Val $\\Delta E$")
    ax1.legend(fontsize=6); ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    p1 = os.path.join(out_dir, "fig_5_1_training_deltae.pdf")
    fig1.savefig(p1, bbox_inches='tight', dpi=300)
    plt.close(fig1)
    print(f"Saved: {p1}")

    fig2, ax2 = plt.subplots(1, 1, figsize=(4.5, 2.8))
    for label, df_h in hist_data.items():
        c = colors_map.get(label, None)
        if "val_psnr" in df_h.columns:
            ax2.plot(df_h["epoch"], df_h["val_psnr"], label=label, color=c)
            ax2.scatter(df_h["epoch"].iloc[-1], df_h["val_psnr"].iloc[-1],
                        marker='o', s=25, color=c, zorder=5)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val PSNR")
    ax2.legend(fontsize=6); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    p2 = os.path.join(out_dir, "fig_5_2_training_psnr.pdf")
    fig2.savefig(p2, bbox_inches='tight', dpi=300)
    plt.close(fig2)
    print(f"Saved: {p2}")


def fig_caption_examples(test_paths, test_caps, cfg, out_dir, n_examples: int = 10):
    if not test_caps:
        print("No captions available -- skipping caption examples figure")
        return
    S = cfg["img_size"]
    sample_caps = test_caps[:n_examples]
    n_per_row = 5
    n_rows = (len(sample_caps) + n_per_row - 1) // n_per_row

    fig, axes = plt.subplots(n_rows, n_per_row, figsize=(2.8 * n_per_row, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for idx_in_grid, (img_idx, raw_cap, tags) in enumerate(sample_caps[:n_rows * n_per_row]):
        r, c = idx_in_grid // n_per_row, idx_in_grid % n_per_row
        img = Image.open(test_paths[img_idx]).convert("RGB").resize((S, S))
        gray = np.array(img.convert("L"))
        axes[r, c].imshow(gray, cmap="gray")
        axes[r, c].axis("off")
        wrapped_raw  = raw_cap[:45] + ("..." if len(raw_cap) > 45 else "")
        wrapped_tags = tags[:40]    + ("..." if len(tags)    > 40 else "")
        axes[r, c].text(0.5, -0.04, f"Caption: {wrapped_raw}",
                        transform=axes[r, c].transAxes, ha='center', va='top',
                        fontsize=5.5, color='#333', style='italic')
        axes[r, c].text(0.5, -0.12, f"Tags: {wrapped_tags}",
                        transform=axes[r, c].transAxes, ha='center', va='top',
                        fontsize=5.5, color='#0066cc', weight='bold')

    plt.suptitle("BLIP Captions (from grayscale) and Extracted Object Tags", fontsize=9, y=1.01)
    plt.subplots_adjust(wspace=0.05, hspace=0.25)
    pdf_path = os.path.join(out_dir, "fig_6_caption_examples.pdf")
    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved: {pdf_path}")


def fig_architecture(out_dir):
    """Schematic of the text-conditioned U-Net."""
    from matplotlib.patches import FancyBboxPatch  # noqa: F401  (used via add_patch)

    fig, ax = plt.subplots(1, 1, figsize=(7.5, 3.2))
    ax.set_xlim(-0.5, 11.5); ax.set_ylim(-0.5, 3.8); ax.axis('off')

    def box(x, y, w, h, label, color='#dbe9f6', fontsize=6, sublabel=None):
        ax.add_patch(plt.matplotlib.patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05",
            facecolor=color, edgecolor='#333', linewidth=0.6))
        ax.text(x + w / 2, y + h / 2 + (0.08 if sublabel else 0), label,
                ha='center', va='center', fontsize=fontsize, weight='bold')
        if sublabel:
            ax.text(x + w / 2, y + h / 2 - 0.15, sublabel,
                    ha='center', va='center', fontsize=4.5, color='#555')

    def arrow(x1, y1, x2, y2, color='#333'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=color, lw=0.8))

    BLUE = '#dbe9f6'; LBLUE = '#a8d0f0'
    box(0, 2.3, 1.0, 0.8, 'L input', LBLUE, sublabel='(B,1,256,256)')
    arrow(1.0, 2.7, 1.5, 2.7)

    enc_labels = [('Enc1', '64'), ('Enc2', '128'), ('Enc3', '256'), ('Enc4', '512')]
    for i, (name, ch) in enumerate(enc_labels):
        x = 1.5 + i * 1.2
        box(x, 2.3, 0.9, 0.8, name, BLUE, sublabel=f'ch={ch}')
        if i < 3:
            arrow(x + 0.9, 2.7, x + 1.2, 2.7)

    arrow(6.3, 2.7, 6.7, 2.7)
    GREEN = '#d5f0d5'
    dec_labels = [('Dec1', '256'), ('Dec2', '128'), ('Dec3', '64')]
    for i, (name, ch) in enumerate(dec_labels):
        x = 6.7 + i * 1.2
        box(x, 2.3, 0.9, 0.8, name, GREEN, sublabel=f'ch={ch}')
        if i < 2:
            arrow(x + 0.9, 2.7, x + 1.2, 2.7)

    for i in range(3):
        ex = 1.5 + (2 - i) * 1.2 + 0.45
        dx = 6.7 + i * 1.2 + 0.45
        ax.annotate('', xy=(dx, 3.15), xytext=(ex, 3.15),
                    arrowprops=dict(arrowstyle='->', color='#888', lw=0.5,
                                    connectionstyle='arc3,rad=-0.15', linestyle='dashed'))

    ORANGE = '#fde8d0'
    box(7.3, 1.2, 1.8, 0.7, 'TextCondBlock', ORANGE, fontsize=6, sublabel='CrossAttn + FiLM')
    arrow(7.9, 1.9, 7.9, 2.3, '#cc7722')
    arrow(8.5, 1.9, 8.5, 2.3, '#cc7722')

    box(9.9, 2.3, 1.2, 0.8, 'Output', '#e8e8e8', sublabel='ab (2ch)\nor MDN (25ch)')
    arrow(9.5, 2.7, 9.9, 2.7)

    TGREEN = '#d0f0d0'
    box(0.2, 0.1, 1.3, 0.6, 'BLIP', TGREEN, sublabel='caption gen')
    arrow(1.5, 0.4, 2.0, 0.4, '#228822')
    box(2.0, 0.1, 1.3, 0.6, 'caption\n-> tags', TGREEN, fontsize=6)
    arrow(3.3, 0.4, 3.8, 0.4, '#228822')
    box(3.8, 0.1, 1.6, 0.6, 'CLIP ViT-B/32', TGREEN, sublabel='512-dim tokens')
    arrow(5.4, 0.4, 7.3, 0.4, '#228822')
    arrow(7.3, 0.7, 7.3, 1.2, '#228822')

    ax.text(0.5, 1.8,  '(B,1,H,W)',  fontsize=4.5, ha='center', color='#666')
    ax.text(10.5, 1.8, '(B,2,H,W)',  fontsize=4.5, ha='center', color='#666')
    ax.text(5.4, 0.0,  '(B,20,512)', fontsize=4.5, ha='left', color='#228822')
    ax.text(5.5, 3.65, 'Text-Conditioned U-Net Architecture', ha='center',
            fontsize=9, weight='bold')

    pdf_path = os.path.join(out_dir, "fig_1_architecture.pdf")
    fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved: {pdf_path}")


def print_outputs_summary(out_dir):
    print("=" * 60)
    print("OUTPUTS SUMMARY")
    print("=" * 60)
    print(f"\nAll files saved to: {out_dir}\n")

    import glob
    for ext, label in [("*.tex", "TEX tables"), ("*.csv", "CSV data"), ("*.pdf", "PDF figures")]:
        files = sorted(glob.glob(os.path.join(out_dir, ext)))
        if not files:
            continue
        print(f"  {label}:")
        for f in files:
            size = os.path.getsize(f)
            print(f"    {os.path.basename(f):40s} ({size/1024:.1f} KB)")
        print()


# ── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Generate figures and tables.")
    p.add_argument("--n-images", type=int, default=-1,
                   help="Limit dataset size when reproducing the split.")
    p.add_argument("--eval-max-batches", type=int, default=None,
                   help="Override CONFIG['eval_max_batches'].")
    p.add_argument("--no-mount-drive", action="store_true",
                   help="Skip Google Drive mount on Colab.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = default_config(n_images=args.n_images, mount_drive=not args.no_mount_drive)
    if args.eval_max_batches is not None:
        cfg["eval_max_batches"] = args.eval_max_batches

    seed_all(cfg["seed"])
    setup_visual_plotting()

    out_dir = cfg["visuals_dir"]
    os.makedirs(out_dir, exist_ok=True)
    print(f"Device: {cfg['device']}")
    print(f"Results from: {cfg['save_dir']}")
    print(f"Outputs to:   {out_dir}")

    # 1. Reproduce the same split used at training time
    ensure_images(cfg["images_dir"])
    image_paths = list_images(cfg["images_dir"])
    assert image_paths, f"No images found in {cfg['images_dir']}"
    if cfg["n_images"] > 0 and cfg["n_images"] < len(image_paths):
        image_paths = image_paths[-cfg["n_images"]:]
    train_paths, val_paths, test_paths = make_or_load_split(image_paths, cfg)
    print(f"Train:{len(train_paths)} Val:{len(val_paths)} Test:{len(test_paths)}")

    datasets, plain_loaders = make_plain_loaders(
        train_paths, val_paths, test_paths, cfg, train_aug=False
    )
    _, _, test_loader = plain_loaders

    # Only the test split needs text caches for generated outputs.
    test_tc, test_caps = build_text_cache(test_paths, "test", cfg["device"], cfg)
    print(f"Text embeddings: test={len(test_tc)}")

    # Need text loaders only for test. Use empty caches for train/val to satisfy API.
    _, _, te_text_loader = make_text_loaders(datasets, ({}, {}, test_tc), cfg)

    # 2. Load checkpoints
    R = load_experiments(cfg["save_dir"], cfg["device"], cfg["mdn_K"])
    if not R:
        raise RuntimeError("No experiments loaded -- run `python -m source.train` first.")

    lpips_fn = lpips.LPIPS(net='alex').to(cfg["device"])
    fill_metrics(R, te_text_loader, test_loader, cfg["device"], cfg, lpips_fn)

    # 3. Tables
    write_table_main(R, out_dir)
    cat_idx = write_table_per_category(R, test_paths, test_caps, test_tc, cfg["device"], cfg, out_dir) or {}
    write_table_mdn_decoding(R, te_text_loader, cfg["device"], cfg, out_dir, lpips_fn)

    # 4. Figures
    fig_architecture(out_dir)
    fig_qualitative_grid(R, test_paths, test_tc, cat_idx, cfg["device"], cfg, out_dir)
    fig_chroma_recovery(R, te_text_loader, test_tc, len(test_paths), cfg["device"], cfg, out_dir)
    fig_mdn_diversity(R, te_text_loader, cfg["device"], out_dir)
    fig_training_curves(cfg["save_dir"], out_dir)
    fig_caption_examples(test_paths, test_caps, cfg, out_dir)

    print_outputs_summary(out_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
