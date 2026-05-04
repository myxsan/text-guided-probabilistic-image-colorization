"""Model definitions: text-conditioned U-Net colorizer + PatchGAN discriminator.

Two generators:
  * UNetColorizer        -- deterministic baseline, no text input.
  * UNetTextColorizer    -- the same backbone with text conditioning blocks
                            inserted at multiple decoder scales.

Both can output either 2-channel `ab` (mode="reg") or MDN parameters
(mode="mdn", 5K channels for K mixture components).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextConditionedBlock(nn.Module):
    """Token-level cross-attention + (optional) FiLM at one spatial scale.

    Projects 512-d CLIP tokens internally to the channel dim of the feature map.
    """

    def __init__(self, ch: int, text_in_dim: int = 512, n_heads: int = 4,
                 use_film: bool = True):
        super().__init__()
        self.use_film = use_film
        self.norm = nn.GroupNorm(8, ch)
        self.q = nn.Conv2d(ch, ch, 1)
        self.text_proj = nn.Linear(text_in_dim, ch * 2)  # text -> K, V
        self.attn = nn.MultiheadAttention(ch, n_heads, batch_first=True)
        self.out = nn.Conv2d(ch, ch, 1)
        self.ln = nn.LayerNorm(ch)
        if use_film:
            self.fg = nn.Linear(text_in_dim, ch)
            self.fb = nn.Linear(text_in_dim, ch)
            # near-identity init: gamma=0, beta=0 -> h unchanged at start
            nn.init.zeros_(self.fg.weight); nn.init.zeros_(self.fg.bias)
            nn.init.zeros_(self.fb.weight); nn.init.zeros_(self.fb.bias)

    def forward(self, h, text_tokens, text_mask=None):
        B, C, H, W = h.shape
        q = self.q(self.norm(h)).view(B, C, H * W).permute(0, 2, 1)
        kv = self.text_proj(text_tokens)
        k, v = kv.chunk(2, -1)
        key_padding_mask = (~text_mask.bool()) if text_mask is not None else None
        a, _ = self.attn(q, k, v, key_padding_mask=key_padding_mask)
        a = self.ln(a).permute(0, 2, 1).view(B, C, H, W)
        h = h + self.out(a)

        if self.use_film:
            if text_mask is not None:
                mf = text_mask.float().unsqueeze(-1)
                tg = (text_tokens * mf).sum(1) / mf.sum(1).clamp(min=1)
            else:
                tg = text_tokens.mean(1)
            g = self.fg(tg).unsqueeze(-1).unsqueeze(-1)
            b = self.fb(tg).unsqueeze(-1).unsqueeze(-1)
            h = h * (1 + g) + b
        return h


class UNetColorizer(nn.Module):
    """Deterministic U-Net, no text conditioning. The primary baseline."""

    def __init__(self, K: int = 5, mode: str = "reg"):
        super().__init__()
        self.mode, self.K = mode, K
        self.enc1 = nn.Conv2d(1, 64, 4, 2, 1)
        self.enc2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.enc3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.enc4 = nn.Conv2d(256, 512, 4, 2, 1)
        self.up1 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.up2 = nn.ConvTranspose2d(512, 128, 4, 2, 1)
        self.up3 = nn.ConvTranspose2d(256, 64, 4, 2, 1)
        out_ch = 2 if mode == "reg" else 5 * K
        self.up4 = nn.ConvTranspose2d(128, out_ch, 4, 2, 1)

    def forward(self, L, text_tokens=None, text_mask=None, use_text: bool = False):
        e1 = F.leaky_relu(self.enc1(L), .2)
        e2 = F.leaky_relu(self.enc2(e1), .2)
        e3 = F.leaky_relu(self.enc3(e2), .2)
        e4 = F.leaky_relu(self.enc4(e3), .2)
        d1 = F.relu(self.up1(e4))
        d2 = F.relu(self.up2(torch.cat([d1, e3], 1)))
        d3 = F.relu(self.up3(torch.cat([d2, e2], 1)))
        out = self.up4(torch.cat([d3, e1], 1))
        z = torch.zeros(L.size(0), 1, 1, 1, device=L.device)
        if self.mode == "reg":
            return {"ab": torch.tanh(out), "mu": z, "logvar": z}
        B, C, H, W = out.shape
        K = self.K
        return {
            "pi": out[:, :K],
            "mu_mdn": torch.tanh(out[:, K:3 * K].view(B, K, 2, H, W)),
            "log_sigma": out[:, 3 * K:].view(B, K, 2, H, W),
            "mu": z, "logvar": z,
        }


class UNetTextColorizer(nn.Module):
    """U-Net backbone + multi-scale text conditioning (token attention + FiLM).

    text_mode: "global" | "token_attn" | "multiscale" | "multiscale_film"
    """

    def __init__(self, text_in_dim: int = 512, K: int = 5,
                 mode: str = "reg", text_mode: str = "multiscale_film"):
        super().__init__()
        self.mode, self.K, self.text_mode = mode, K, text_mode

        self.enc1 = nn.Conv2d(1, 64, 4, 2, 1)
        self.enc2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.enc3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.enc4 = nn.Conv2d(256, 512, 4, 2, 1)
        self.up1 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.up2 = nn.ConvTranspose2d(512, 128, 4, 2, 1)
        self.up3 = nn.ConvTranspose2d(256, 64, 4, 2, 1)

        # CLIP tokens are 512-d; project only when caller wants something else.
        self.clip_proj = nn.Linear(512, text_in_dim) if text_in_dim != 512 else nn.Identity()

        use_film = text_mode in ("multiscale_film", "global")
        if text_mode in ("multiscale", "multiscale_film"):
            self.t1 = TextConditionedBlock(256, text_in_dim, use_film=use_film)
            self.t2 = TextConditionedBlock(128, text_in_dim, use_film=use_film)
            self.t3 = TextConditionedBlock(64,  text_in_dim, use_film=use_film)
        elif text_mode == "token_attn":
            self.t1 = TextConditionedBlock(256, text_in_dim, use_film=False)
        elif text_mode == "global":
            for ch, name in [(256, "f1"), (128, "f2"), (64, "f3")]:
                f = nn.ModuleDict({
                    "g": nn.Linear(text_in_dim, ch),
                    "b": nn.Linear(text_in_dim, ch),
                })
                nn.init.zeros_(f["g"].weight); nn.init.zeros_(f["g"].bias)
                nn.init.zeros_(f["b"].weight); nn.init.zeros_(f["b"].bias)
                setattr(self, name, f)

        out_ch = 2 if mode == "reg" else 5 * K
        self.up4 = nn.ConvTranspose2d(128, out_ch, 4, 2, 1)

    def _global_text(self, text_tokens, text_mask):
        if text_mask is not None:
            mf = text_mask.float().unsqueeze(-1)
            return (text_tokens * mf).sum(1) / mf.sum(1).clamp(min=1)
        return text_tokens.mean(1)

    def _apply_film(self, h, film, tg):
        g = film["g"](tg).unsqueeze(-1).unsqueeze(-1)
        b = film["b"](tg).unsqueeze(-1).unsqueeze(-1)
        return h * (1 + g) + b

    def forward(self, L, text_tokens=None, text_mask=None, use_text: bool = False):
        ht = use_text and text_tokens is not None
        if ht:
            text_tokens = self.clip_proj(text_tokens)

        e1 = F.leaky_relu(self.enc1(L), .2)
        e2 = F.leaky_relu(self.enc2(e1), .2)
        e3 = F.leaky_relu(self.enc3(e2), .2)
        e4 = F.leaky_relu(self.enc4(e3), .2)

        d1 = F.relu(self.up1(e4))
        if ht:
            if self.text_mode in ("multiscale", "multiscale_film", "token_attn"):
                d1 = self.t1(d1, text_tokens, text_mask)
            elif self.text_mode == "global":
                d1 = self._apply_film(d1, self.f1, self._global_text(text_tokens, text_mask))

        d2 = F.relu(self.up2(torch.cat([d1, e3], 1)))
        if ht and self.text_mode in ("multiscale", "multiscale_film"):
            d2 = self.t2(d2, text_tokens, text_mask)
        elif ht and self.text_mode == "global":
            d2 = self._apply_film(d2, self.f2, self._global_text(text_tokens, text_mask))

        d3 = F.relu(self.up3(torch.cat([d2, e2], 1)))
        if ht and self.text_mode in ("multiscale", "multiscale_film"):
            d3 = self.t3(d3, text_tokens, text_mask)
        elif ht and self.text_mode == "global":
            d3 = self._apply_film(d3, self.f3, self._global_text(text_tokens, text_mask))

        out = self.up4(torch.cat([d3, e1], 1))
        z = torch.zeros(L.size(0), 1, 1, 1, device=L.device)
        if self.mode == "reg":
            return {"ab": torch.tanh(out), "mu": z, "logvar": z}
        B, C, H, W = out.shape
        K = self.K
        return {
            "pi": out[:, :K],
            "mu_mdn": torch.tanh(out[:, K:3 * K].view(B, K, 2, H, W)),
            "log_sigma": out[:, 3 * K:].view(B, K, 2, H, W),
            "mu": z, "logvar": z,
        }


class PatchGAN(nn.Module):
    """Standard PatchGAN discriminator over (L, RGB) pairs."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(4, 64, 4, 2, 1)),  nn.LeakyReLU(.2),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)), nn.LeakyReLU(.2),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)), nn.LeakyReLU(.2),
            nn.Conv2d(256, 1, 4, 1, 1),
        )

    def forward(self, L, rgb):
        return self.net(torch.cat([L, rgb], 1))


def build_generator(use_text: bool, mode: str = "reg",
                    text_mode: str = "token_attn", K: int = 5,
                    text_in_dim: int = 512):
    """Pick the right generator class given training mode and text usage."""
    if not use_text:
        return UNetColorizer(K=K, mode=mode)
    return UNetTextColorizer(text_in_dim=text_in_dim, K=K, mode=mode, text_mode=text_mode)
