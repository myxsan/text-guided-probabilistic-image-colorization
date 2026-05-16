# UI

Streamlit demo for the trained text-guided colorization models.

## Run

```bash
pip install streamlit
streamlit run ui/app.py
```

(`torch`, `transformers`, `Pillow`, `scikit-image`, and `numpy` from
`source/requirements.txt` must also be installed.)

The app auto-loads checkpoints from `results/<experiment>/best_val_checkpoint.pt`.
Pick the model from the dropdown — `ours_mdn` is the default. Upload any image;
it is forced to grayscale, resized to 256×256, captioned with BLIP, and
colorized.
