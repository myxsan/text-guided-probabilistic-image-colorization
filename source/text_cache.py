"""Caption -> tag -> CLIP token-embedding pipeline.

For each image we feed its grayscale version to BLIP, prune the resulting
caption down to a small bag of object/scene nouns (`caption_to_tags`), and
then encode that string with CLIP's text tower. The token-level embeddings
are what the U-Net's cross-attention modules consume.
"""

import os
import re

import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    CLIPTokenizer, CLIPTextModel,
)

# Stop-words: kept aggressive to drop verbs/adjectives/article noise from
# generated captions and leave only object/scene nouns.
_STOP = set((
    "a an the of in on at to and with is are was were that this for it its by from "
    "as or be been being do does did has have had will would shall should may might can could "
    "not no nor so if but very much more most also just only even still already yet than "
    "some any all each every both few many several own other another such "
    "here there where when how what which who whom whose why "
    "black white photo photograph image picture view shown camera shot scene background "
    "foreground side front back top bottom left right center middle large small big little "
    "old new first last next long short high low good bad great real close open going sitting "
    "standing looking wearing holding playing walking running making taking getting using "
    "two three four five group pair set lot number kind type part piece bit way thing "
    "day room area place point end line made seen "
    "between above below behind along across through around near "
    "young different same full empty various about away down into "
    "eating flying lying parked covered filled moving turned during"
).split())


def caption_to_tags(caption: str) -> str:
    """Return up to 8 content nouns from a caption as a comma-separated string."""
    words = re.findall(r"[a-z]+", caption.lower())
    kept = [w for w in words if w not in _STOP and len(w) > 2][:8]
    return ", ".join(kept) or caption


@torch.no_grad()
def build_text_cache(paths, split_name: str, device: str, cfg: dict):
    """For each image: BLIP caption -> tags -> CLIP token embeddings.

    Returns
    -------
    cache : dict[int, {"tokens": Tensor[T, 512], "mask": Tensor[T]}]
        Token-level CLIP encodings keyed by index into `paths`.
    caps  : list[(idx, raw_caption, tag_string)]
        Diagnostic info (only populated when we recompute, empty when loaded
        from an on-disk cache).
    """
    cache_path = os.path.join(cfg["cache_dir"], f"{split_name}_text_tokens.pt")
    if cfg.get("cache_text") and os.path.exists(cache_path):
        print(f"Loading cached text: {cache_path}")
        return torch.load(cache_path, map_location="cpu"), []

    print(f"Computing text embeddings for {split_name} ({len(paths)} images)...")
    blip_proc = BlipProcessor.from_pretrained(cfg["caption_model"])
    blip = BlipForConditionalGeneration.from_pretrained(cfg["caption_model"]).to(device).eval()
    clip_tok = CLIPTokenizer.from_pretrained(cfg["clip_text_model"])
    clip_txt = CLIPTextModel.from_pretrained(cfg["clip_text_model"]).to(device).eval()

    cache, caps = {}, []
    for i in tqdm(range(len(paths)), desc=f"Caption {split_name}"):
        # Feed BLIP the grayscale version so it sees what the colorizer sees.
        gray = Image.open(paths[i]).convert("RGB").convert("L").convert("RGB")
        inp = blip_proc(images=gray, return_tensors="pt").to(device)
        ids = blip.generate(**inp, max_new_tokens=cfg["caption_max_new_tokens"])
        raw = blip_proc.decode(ids[0], skip_special_tokens=True).strip()
        tags = caption_to_tags(raw)
        caps.append((i, raw, tags))

        tok = clip_tok(
            [tags], padding="max_length", truncation=True,
            max_length=cfg["max_text_tokens"], return_tensors="pt",
        ).to(device)
        out = clip_txt(**tok)
        cache[i] = {
            "tokens": out.last_hidden_state.squeeze(0).cpu().float(),
            "mask": tok["attention_mask"].squeeze(0).cpu(),
        }

    if cfg.get("cache_text"):
        torch.save(cache, cache_path)
        print(f"Saved: {cache_path}")
    return cache, caps
