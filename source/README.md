# source/

Modular Python port of the two notebooks at the project root:

| Notebook              | Equivalent here              |
|-----------------------|------------------------------|
| `colab_notebook.ipynb`| `python -m source.train`     |
| `test_outputs.ipynb`  | `python -m source.generate_outputs` |

## Layout

```
source/
├── config.py            CONFIG dict, env-aware paths (Colab + local), seed_all
├── colorlab.py          Lab/RGB conversions, CIEDE2000, MDN helpers, chroma loss
├── models.py            TextConditionedBlock, UNetColorizer, UNetTextColorizer, PatchGAN
├── text_cache.py        BLIP captions -> tag bag -> CLIP token embeddings
├── data.py              Datasets, splits, DataLoaders, batch unpacking
├── dataset_setup.py     Kagglehub download or use existing local images
├── train.py             4-experiment ablation runner (entry point)
└── generate_outputs.py  Generated figures and TEX tables (entry point)
```

## Setup

```bash
pip install -r source/requirements.txt
```

## Inputs / outputs

Defaults assume the project layout shown below. You can override any of them
with environment variables:

| Variable           | Local default               | Colab default                                          |
|--------------------|-----------------------------|--------------------------------------------------------|
| `IMAGES_DIR`       | `./data/coco_images`        | `/content/coco_local`                                  |
| `RESULTS_DIR`      | `./results`                 | `/content/drive/My Drive/colorization_runs_coco2017`   |
| `CACHE_DIR`        | `./cache`                   | `/content/drive/My Drive/colorization_cache_coco2017`  |
| `VISUALS_DIR`      | `./visuals`                 | `/content/visuals`                                     |

If `IMAGES_DIR` is empty, both scripts pull
`abdelrahmanelgharibx/coco2017-subset` from Kaggle via `kagglehub` and flatten
all images into the directory.

## Usage

Train all four experiments (baseline, ours, ours+chroma, ours+MDN):

```bash
python -m source.train
# subset for a quick smoke test:
python -m source.train --n-images 1000 --epochs 3
# only re-run a subset:
python -m source.train --experiments ours_mdn
```

Generate figures and tables from the saved checkpoints:

```bash
python -m source.generate_outputs
# faster pass (fewer test batches for MDN decoding eval):
python -m source.generate_outputs --eval-max-batches 20
```

## Colab usage

The scripts also work as-is from a Colab cell. Just clone the repo and run:

```python
!git clone https://github.com/<user>/<repo>.git
%cd <repo>
!pip -q install -r source/requirements.txt
!python -m source.train
!python -m source.generate_outputs
```

The Drive mount is attempted automatically. Pass `--no-mount-drive` (or set
`RESULTS_DIR` / `CACHE_DIR` to local paths) to skip it.
