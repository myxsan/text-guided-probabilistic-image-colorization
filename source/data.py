"""Dataset, splits, DataLoaders.

`CDS` returns plain (L, ab, rgb, idx) tuples. `TDS` wraps a CDS and additionally
emits per-image text tokens + attention mask for text-conditioned training.
"""

import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from skimage import color as skcolor

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def list_images(images_dir: str):
    return sorted([
        os.path.join(images_dir, f) for f in os.listdir(images_dir)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    ])


def make_or_load_split(image_paths, cfg):
    """Train/val/test split that is stable across runs (seeded + cached on disk).

    The cache file is keyed by `n_images` so different subset sizes get
    different splits, and we re-generate when the on-disk indices no longer
    match the available files.
    """
    cache_tag = "all" if cfg["n_images"] == -1 else str(cfg["n_images"])
    split_file = os.path.join(cfg["cache_dir"], f"split_indices_n{cache_tag}.json")

    use_cached = False
    if os.path.exists(split_file):
        try:
            with open(split_file) as f:
                sd = json.load(f)
            train_idx, val_idx, test_idx = sd["train"], sd["val"], sd["test"]
            all_idx = train_idx + val_idx + test_idx
            if all_idx and max(all_idx) < len(image_paths):
                use_cached = True
                print(f"Loaded cached split: {split_file}")
            else:
                print(f"Cached split invalid for current files. Regenerating: {split_file}")
        except Exception:
            print(f"Could not read split cache. Regenerating: {split_file}")

    if not use_cached:
        random.seed(cfg["seed"])
        idx = list(range(len(image_paths)))
        random.shuffle(idx)
        n = len(idx)
        nt = int(n * cfg["train_ratio"])
        nv = int(n * cfg["val_ratio"])
        train_idx = sorted(idx[:nt])
        val_idx   = sorted(idx[nt:nt + nv])
        test_idx  = sorted(idx[nt + nv:])
        with open(split_file, "w") as f:
            json.dump({
                "seed": cfg["seed"], "n_images": cfg["n_images"],
                "train": train_idx, "val": val_idx, "test": test_idx,
            }, f)
        print(f"Saved split cache: {split_file}")

    train_paths = [image_paths[i] for i in train_idx]
    val_paths   = [image_paths[i] for i in val_idx]
    test_paths  = [image_paths[i] for i in test_idx]
    return train_paths, val_paths, test_paths


class CDS(Dataset):
    """Plain colorization dataset: returns (L, ab, rgb_full, idx)."""

    def __init__(self, paths, transform):
        self.paths, self.tfm = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = self.tfm(Image.open(self.paths[i]).convert("RGB"))
        rgb = np.array(img).astype(np.float32) / 255
        lab = skcolor.rgb2lab(rgb)
        L  = torch.from_numpy(lab[..., 0:1] / 100).permute(2, 0, 1).float()
        ab = torch.from_numpy(lab[..., 1:3] / 128).permute(2, 0, 1).float()
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float()
        return L, ab, rgb_t, i


class TDS(Dataset):
    """Wraps a CDS and adds per-image text tokens (CLIP) + attention mask."""

    def __init__(self, base, cache, text_dim, max_tokens):
        self.base, self.cache = base, cache
        self.text_dim, self.max_tokens = text_dim, max_tokens

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        L, ab, rgb, idx = self.base[i]
        e = self.cache.get(int(idx))
        if e is not None:
            t = e["tokens"][:self.max_tokens]
            m = e["mask"][:self.max_tokens]
        else:
            t = torch.zeros(1, self.text_dim)
            m = torch.zeros(1, dtype=torch.long)
        return L, ab, rgb, idx, t, m


def text_collate(batch):
    """Pad variable-length token sequences to the longest in the batch."""
    Ls, As, Rs, Is, Ts, Ms = zip(*batch)
    mT = max(t.size(0) for t in Ts)
    D = Ts[0].size(1)
    tp = torch.zeros(len(batch), mT, D)
    mp = torch.zeros(len(batch), mT, dtype=torch.long)
    for i, (t, m) in enumerate(zip(Ts, Ms)):
        T = t.size(0)
        tp[i, :T] = t
        mp[i, :T] = m
    return (torch.stack(Ls), torch.stack(As), torch.stack(Rs),
            torch.tensor(Is), tp, mp)


def make_transforms(img_size: int):
    """Train transform = resize + flip + light color jitter; eval = resize only."""
    train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.3),
    ])
    eval_ = transforms.Compose([transforms.Resize((img_size, img_size))])
    return train, eval_


def make_plain_loaders(train_paths, val_paths, test_paths, cfg, train_aug: bool = True):
    train_tfm, eval_tfm = make_transforms(cfg["img_size"])
    train_ds = CDS(train_paths, train_tfm if train_aug else eval_tfm)
    val_ds   = CDS(val_paths,   eval_tfm)
    test_ds  = CDS(test_paths,  eval_tfm)
    common = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"], pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True,  **common)
    val_loader   = DataLoader(val_ds,   shuffle=False, **common)
    test_loader  = DataLoader(test_ds,  shuffle=False, **common)
    return (train_ds, val_ds, test_ds), (train_loader, val_loader, test_loader)


def make_text_loaders(datasets, text_caches, cfg):
    """Wrap (train, val, test) CDS with TDS+collate so batches carry text tokens."""
    train_ds, val_ds, test_ds = datasets
    tc_train, tc_val, tc_test = text_caches
    MT = cfg["max_text_tokens"]
    tr = TDS(train_ds, tc_train, 512, MT)
    vl = TDS(val_ds,   tc_val,   512, MT)
    te = TDS(test_ds,  tc_test,  512, MT)
    common = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
                  pin_memory=True, collate_fn=text_collate)
    return (
        DataLoader(tr, shuffle=True,  **common),
        DataLoader(vl, shuffle=False, **common),
        DataLoader(te, shuffle=False, **common),
    )


def unpack_batch(batch, use_text: bool, dev):
    """Move a batch to device and unpack into a uniform 5-tuple.

    Returns: (L, ab, rgb, text_tokens_or_None, text_mask_or_None)
    """
    if len(batch) == 6:
        L, ab, rgb, _idx, t, m = batch
        L, ab, rgb = L.to(dev), ab.to(dev), rgb.to(dev)
        return L, ab, rgb, (t.to(dev) if use_text else None), (m.to(dev) if use_text else None)
    L, ab, rgb, _idx = batch
    return L.to(dev), ab.to(dev), rgb.to(dev), None, None
