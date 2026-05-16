"""Streamlit demo for the text-guided colorization model.

Loads one of the trained checkpoints from results/, accepts a grayscale image,
runs BLIP captioning -> CLIP text encoding -> U-Net colorization, and shows
the colorized output side by side with the input.
"""

import base64
import json
import os
import sys
from io import BytesIO

import numpy as np
import streamlit as st
import torch
from PIL import Image
from skimage import color as skcolor

# Make `source/` importable when this file is run from inside `ui/`.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from source.colorlab import lab_to_rgb_np, mdn_expected_ab
from source.models import UNetColorizer, UNetTextColorizer
from source.text_cache import caption_to_tags


IMG_SIZE = 256          # model input/output resolution
DISPLAY_SIZE = 256      # on-screen render size for the input/output thumbnails
RESULTS_DIR = os.path.join(ROOT, "results")
DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)

MODEL_CHOICES = {
    "Ours + MDN (probabilistic, vivid)": "ours_mdn",
    "Ours + Chroma loss (best ΔE / FID)": "ours_chroma",
    "Ours (text-conditioned)": "ours",
    "Baseline (no text, no chroma)": "baseline",
}
DEFAULT_MODEL_LABEL = "Ours + MDN (probabilistic, vivid)"


# ── Page setup ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Text-Guided Colorization",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at 20% 0%, #1a1f3a 0%, #0f1224 50%, #060814 100%);
        color: #e6e8ef;
    }
    .block-container {
        padding-top: 4.5rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }
    h1 {
        font-family: "Georgia", "Times New Roman", serif;
        font-weight: 600;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #f5e1c8, #d4a5ff 60%, #8ecaff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-top: 0.5rem;
        margin-bottom: 1.6rem;
    }
    /* Cap displayed images so they never exceed our intended display size. */
    div[data-testid="stImage"] img {
        max-width: 256px !important;
        max-height: 256px !important;
        height: auto;
    }
    /* Streamlit forces width:100% on markdown images, which blows up our
       base64-embedded thumbnails. Pin them to exactly 256x256. */
    [data-testid="stMarkdown"] img.thumb-img,
    img.thumb-img {
        width: 256px !important;
        height: 256px !important;
        max-width: 256px !important;
        max-height: 256px !important;
        flex-shrink: 0 !important;
        display: block !important;
        border-radius: 8px !important;
    }
    .panel {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 1.4rem 1.5rem 1.2rem 1.5rem;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.35);
    }
    .panel-title {
        color: #c9d0ec;
        font-size: 0.8rem;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 0.7rem;
        font-weight: 600;
    }
    .img-frame {
        display: inline-block;
        padding: 6px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
    }
    .caption-pill {
        background: rgba(212, 165, 255, 0.12);
        border: 1px solid rgba(212, 165, 255, 0.3);
        border-radius: 10px;
        padding: 0.55rem 0.85rem;
        color: #e6e0ff;
        font-size: 0.8rem;
        margin-bottom: 0.5rem;
        line-height: 1.45;
        word-break: break-word;
    }
    .caption-pill b, .tag-pill b {
        display: block;
        font-size: 0.7rem;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        opacity: 0.7;
        margin-bottom: 0.25rem;
    }
    .tag-pill {
        background: rgba(142, 202, 255, 0.12);
        border: 1px solid rgba(142, 202, 255, 0.3);
        border-radius: 10px;
        padding: 0.55rem 0.85rem;
        color: #d6ecff;
        font-size: 0.8rem;
        line-height: 1.45;
        word-break: break-word;
    }
    .footer {
        text-align: center;
        color: #6b7395;
        font-size: 0.78rem;
        margin-top: 2.5rem;
        padding-top: 1.2rem;
        border-top: 1px solid rgba(255, 255, 255, 0.06);
    }
    [data-testid="stFileUploader"] section {
        background: rgba(255, 255, 255, 0.03);
        border: 1.5px dashed rgba(212, 165, 255, 0.35);
        border-radius: 12px;
        padding: 0.9rem 1rem;
        min-height: 0;
    }
    [data-testid="stFileUploader"] section:hover {
        border-color: rgba(212, 165, 255, 0.6);
        background: rgba(212, 165, 255, 0.05);
    }
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] small {
        font-size: 0.7rem;
        opacity: 0.6;
    }
    .stSelectbox label {
        color: #9fa6c2 !important;
        font-size: 0.75rem;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }
    div[data-testid="stImage"] img {
        border-radius: 8px;
    }
    [data-testid="stFileUploaderFile"],
    [data-testid="stFileUploaderDeleteBtn"] {
    display: none !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1>Text-Guided Colorization</h1>", unsafe_allow_html=True)


# ── Cached loaders ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading colorization model…")
def load_colorizer(exp_name: str):
    """Instantiate the U-Net for `exp_name` and load its best checkpoint."""
    exp_dir = os.path.join(RESULTS_DIR, exp_name)
    with open(os.path.join(exp_dir, "config.json")) as f:
        exp_cfg = json.load(f)

    mode = exp_cfg.get("mode", "reg")
    text_mode = exp_cfg.get("text_mode", "none")
    use_text = exp_cfg.get("use_text", False) and text_mode != "none"
    K = exp_cfg.get("mdn_K", 5)

    if use_text:
        model = UNetTextColorizer(text_in_dim=512, K=K, mode=mode, text_mode=text_mode)
    else:
        model = UNetColorizer(K=K, mode=mode)

    state = torch.load(
        os.path.join(exp_dir, "best_val_checkpoint.pt"),
        map_location=DEVICE,
        weights_only=False,
    )
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model, {"mode": mode, "use_text": use_text, "text_mode": text_mode}


@st.cache_resource(show_spinner="Loading BLIP captioner + CLIP text encoder…")
def load_text_pipeline():
    """BLIP (caption) + CLIP (tokenizer + text encoder)."""
    from transformers import (
        BlipForConditionalGeneration, BlipProcessor,
        CLIPTextModel, CLIPTokenizer,
    )
    blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(DEVICE).eval()
    clip_tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_txt = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE).eval()
    return blip_proc, blip, clip_tok, clip_txt


# ── Inference ───────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_text(pil_gray_rgb: Image.Image):
    """Caption a grayscale-as-RGB image and return (caption, tags, tokens, mask)."""
    blip_proc, blip, clip_tok, clip_txt = load_text_pipeline()
    inp = blip_proc(images=pil_gray_rgb, return_tensors="pt").to(DEVICE)
    ids = blip.generate(**inp, max_new_tokens=20)
    caption = blip_proc.decode(ids[0], skip_special_tokens=True).strip()
    tags = caption_to_tags(caption)

    tok = clip_tok(
        [tags], padding="max_length", truncation=True,
        max_length=20, return_tensors="pt",
    ).to(DEVICE)
    out = clip_txt(**tok)
    return caption, tags, out.last_hidden_state, tok["attention_mask"]


@torch.no_grad()
def colorize(pil_img: Image.Image, exp_name: str):
    """Run the full pipeline. Returns (rgb_out_uint8, caption, tags, gray_uint8)."""
    # Force grayscale, then back to 3-channel so the same path works for color
    # uploads too (the model only sees L either way).
    gray_rgb = pil_img.convert("L").convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    rgb_np = np.array(gray_rgb).astype(np.float32) / 255.0
    lab = skcolor.rgb2lab(rgb_np)
    L = torch.from_numpy(lab[..., 0:1] / 100.0).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE)

    model, meta = load_colorizer(exp_name)

    caption, tags, tokens, mask = "", "", None, None
    if meta["use_text"]:
        caption, tags, tokens, mask = encode_text(gray_rgb)

    out = model(L, text_tokens=tokens, text_mask=mask, use_text=meta["use_text"])
    ab = out["ab"] if meta["mode"] == "reg" else mdn_expected_ab(out["pi"], out["mu_mdn"])
    rgb_pred = lab_to_rgb_np(L, ab).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    rgb_out = (rgb_pred * 255).astype(np.uint8)

    gray_out = (np.array(gray_rgb.convert("L"))).astype(np.uint8)
    return rgb_out, caption, tags, gray_out


# ── Helpers ────────────────────────────────────────────────────────────────

def img_data_url(img) -> str:
    """PIL image or HxWx3 uint8 ndarray -> data: URL for inline <img> tags."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ── Layout ──────────────────────────────────────────────────────────────────

model_label = st.selectbox(
    "Model",
    list(MODEL_CHOICES.keys()),
    index=list(MODEL_CHOICES.keys()).index(DEFAULT_MODEL_LABEL),
)
exp_name = MODEL_CHOICES[model_label]

uploaded = st.file_uploader(
    "Drop or click to upload a grayscale image",
    type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
    label_visibility="collapsed",
)

st.write("")  # small gap

# Two equal columns: input on the left, output (image + caption pills) on
# the right. Input image is right-aligned and output content is left-aligned
# so the two image squares meet at the centerline of the page; the pills
# extend rightward into the rest of the output column.
col_left, col_right = st.columns(2, gap="medium")

def placeholder_html(text: str) -> str:
    return (
        f"<div style='color:#6b7395;text-align:center;"
        f"border:1.5px dashed rgba(255,255,255,0.08);border-radius:14px;"
        f"width:{DISPLAY_SIZE}px;height:{DISPLAY_SIZE}px;display:flex;"
        f"align-items:center;justify-content:center;font-size:0.85rem;'>"
        f"{text}</div>"
    )

input_img = None
if uploaded is not None:
    input_img = Image.open(uploaded)

with col_left:
    st.markdown(
        f"<div class='panel-title' style='text-align:left;"
        f"width:{DISPLAY_SIZE}px;margin-left:auto;'>"
        f"Input · Grayscale</div>",
        unsafe_allow_html=True,
    )
    if input_img is None:
        st.markdown(
            f"<div style='display:flex;justify-content:flex-end;'>"
            f"{placeholder_html('Waiting for input…')}</div>",
            unsafe_allow_html=True,
        )
    else:
        gray = input_img.convert("L").resize((IMG_SIZE, IMG_SIZE))
        st.markdown(
            f"<div style='display:flex;justify-content:flex-end;'>"
            f"<img class='thumb-img' src='{img_data_url(gray)}' />"
            f"</div>",
            unsafe_allow_html=True,
        )

with col_right:
    st.markdown("<div class='panel-title'>Output · Colorized</div>", unsafe_allow_html=True)
    if input_img is None:
        st.markdown(placeholder_html("Upload an image to see the colorized result."),
                    unsafe_allow_html=True)
    else:
        with st.spinner("Colorizing…"):
            rgb_out, caption, tags, _ = colorize(input_img, exp_name)

        # One HTML block: image and pills side by side via flexbox.
        pills = ""
        if caption:
            pills += f"<div class='caption-pill'><b>Caption</b>{caption}</div>"
        if tags:
            pills += f"<div class='tag-pill'><b>Tags</b>{tags}</div>"
        st.markdown(
            "<div style='display:flex;gap:14px;align-items:flex-start;'>"
            f"  <img class='thumb-img' src='{img_data_url(rgb_out)}' />"
            f"  <div style='flex:1;min-width:0;'>{pills}</div>"
            "</div>",
            unsafe_allow_html=True,
        )