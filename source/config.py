"""Configuration: paths and hyperparameters.

Auto-detects Colab vs local environment. All paths can be overridden via
environment variables so the same code runs unchanged in both places.

Environment variables:
    IMAGES_DIR   -- where COCO images live           (default: ./data/coco_images   | colab: /content/coco_local)
    RESULTS_DIR  -- where training checkpoints land  (default: ./results            | colab: /content/drive/My Drive/colorization_runs_coco2017)
    CACHE_DIR    -- where text/embedding caches land (default: ./cache              | colab: /content/drive/My Drive/colorization_cache_coco2017)
    VISUALS_DIR  -- where generated figures/tables land (default: ./visuals         | colab: /content/visuals)
"""

import os
import random
import numpy as np
import torch


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _try_mount_drive() -> bool:
    """Mount Google Drive if running in Colab. Returns True on success."""
    if not is_colab():
        return False
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        return True
    except Exception:
        return False


def _project_root() -> str:
    """Resolve the project root (one level above this file)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_paths(mount_drive: bool = True) -> dict:
    """Pick sensible default directories for the current environment.

    Local: paths live inside the repo so a fresh clone "just works".
    Colab: image data goes in session storage; checkpoints/cache go to
    Drive when mounted (so they survive runtime resets).
    """
    if is_colab():
        drive_ok = _try_mount_drive() if mount_drive else False
        images_default  = "/content/coco_local"
        results_default = ("/content/drive/My Drive/colorization_runs_coco2017"
                           if drive_ok else "/content/colorization_runs")
        cache_default   = ("/content/drive/My Drive/colorization_cache_coco2017"
                           if drive_ok else "/content/colorization_cache")
        visuals_default = "/content/visuals"
    else:
        root = _project_root()
        images_default  = os.path.join(root, "data", "coco_images")
        results_default = os.path.join(root, "results")
        cache_default   = os.path.join(root, "cache")
        visuals_default = os.path.join(root, "visuals")

    return {
        "images_dir":   os.environ.get("IMAGES_DIR",   images_default),
        "results_dir":  os.environ.get("RESULTS_DIR",  results_default),
        "cache_dir":    os.environ.get("CACHE_DIR",    cache_default),
        "visuals_dir":  os.environ.get("VISUALS_DIR",  visuals_default),
    }


def build_config(images_dir: str, results_dir: str, cache_dir: str,
                 n_images: int = -1) -> dict:
    """Assemble the CONFIG dict used throughout training and evaluation."""
    return {
        # Dataset
        "n_images": n_images,
        "images_dir": images_dir,
        "train_ratio": 0.70, "val_ratio": 0.15, "test_ratio": 0.15, "seed": 42,
        # Optimization
        "img_size": 256, "batch_size": 16, "epochs": 25,
        "lr_g": 2e-4, "lr_d": 2e-4, "lr_g_reg": 2e-4, "lr_g_mdn": 1e-4,
        "lambda_adv": 1.0, "lambda_lpips": 1.0, "lambda_recon": 2.0,
        "gan_warmup_epochs_reg": 3, "gan_warmup_epochs_mdn": 2,
        "lambda_adv_reg": 0.1, "lambda_adv_mdn": 0.1,
        "lambda_recon_reg": 10.0, "lambda_recon_mdn": 5.0,
        # MDN
        "mdn_K": 5, "mdn_min_sigma": 1e-3,
        "mdn_log_sigma_min": -7.0, "mdn_log_sigma_max": 3.0,
        "mdn_temperatures": [0.5, 0.8, 1.0, 1.5], "mdn_best_of_n": 5,
        # Text
        "use_text": True, "text_dim": 512, "max_text_tokens": 20,
        "caption_max_new_tokens": 20,
        "caption_model": "Salesforce/blip-image-captioning-base",
        "clip_text_model": "openai/clip-vit-base-patch32",
        "text_mode": "token_attn",
        # Chroma loss
        "lambda_chroma": 0.1,
        # Caching / I/O
        "cache_text": False,
        "cache_dir": cache_dir,
        "save_dir":  results_dir,
        # Training control
        "early_stopping": True, "es_patience": 5, "es_min_delta": 1e-4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "num_workers": 0, "clip_grad": 1.0, "eval_max_batches": 50,
    }


def default_config(n_images: int = -1, mount_drive: bool = True) -> dict:
    """Convenience: resolve paths + build CONFIG + create directories."""
    paths = resolve_paths(mount_drive=mount_drive)
    cfg = build_config(
        images_dir=paths["images_dir"],
        results_dir=paths["results_dir"],
        cache_dir=paths["cache_dir"],
        n_images=n_images,
    )
    cfg["visuals_dir"] = paths["visuals_dir"]
    os.makedirs(cfg["cache_dir"], exist_ok=True)
    os.makedirs(cfg["save_dir"],  exist_ok=True)
    return cfg


def seed_all(s: int = 42) -> None:
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
