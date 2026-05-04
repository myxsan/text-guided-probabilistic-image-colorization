"""Make sure `images_dir` is populated with COCO-2017 images.

If the directory already has images, do nothing. Otherwise pull
`abdelrahmanelgharibx/coco2017-subset` via kagglehub, unzip, and flatten
all images into `images_dir`. Useful both on Colab (where data isn't
persistent) and on a fresh local clone.
"""

import glob
import os
import shutil
import zipfile

from .data import IMG_EXTS

KAGGLE_DS = "abdelrahmanelgharibx/coco2017-subset"


def _flatten_images(src_root: str, dst_dir: str) -> int:
    """Copy/hardlink every image found under `src_root` into `dst_dir`."""
    count = 0
    for root, _, files in os.walk(src_root):
        for fn in files:
            if os.path.splitext(fn)[1].lower() not in IMG_EXTS:
                continue
            src = os.path.join(root, fn)
            dst = os.path.join(dst_dir, fn)
            if os.path.exists(dst):
                continue
            try:
                same_device = os.stat(src).st_dev == os.stat(dst_dir).st_dev
            except FileNotFoundError:
                same_device = False
            if same_device:
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)
            else:
                shutil.copy2(src, dst)
            count += 1
    return count


def ensure_images(images_dir: str) -> str:
    """Guarantee `images_dir` contains image files; returns `images_dir`."""
    os.makedirs(images_dir, exist_ok=True)
    existing = [f for f in os.listdir(images_dir)
                if os.path.splitext(f)[1].lower() in IMG_EXTS]
    if existing:
        print(f"Dataset already present at {images_dir} ({len(existing)} images).")
        return images_dir

    try:
        import kagglehub
    except ImportError as e:
        raise RuntimeError(
            f"No images in {images_dir} and kagglehub is not installed. "
            "Install kagglehub (`pip install kagglehub`) or place images in the "
            "directory manually."
        ) from e

    dl_path = kagglehub.dataset_download(KAGGLE_DS)
    print(f"Downloaded to: {dl_path}")

    # Some Kaggle datasets ship as zips inside the download dir.
    for zf in glob.glob(os.path.join(dl_path, "**", "*.zip"), recursive=True):
        print(f"Extracting {zf}...")
        with zipfile.ZipFile(zf) as z:
            z.extractall(dl_path)

    n = _flatten_images(dl_path, images_dir)
    print(f"Flattened {n} images into {images_dir}")
    return images_dir
