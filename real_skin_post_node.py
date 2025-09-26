# skin_fix_by_noise_node.py
# ComfyUI custom node: RealSkinPost (Simple) + (Advanced)
# Adds realistic microtexture/pores to skin as a post-process.
# No changes to your WAN 2.2 workflow required—just wire this after your final image.
#
# Dependencies: Pillow, numpy, torch (already present with ComfyUI)

import math
import random
import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps

# ---------------------------
# Utilities
# ---------------------------

def _tensor_to_np(img_t: torch.Tensor) -> np.ndarray:
    """
    Convert ComfyUI IMAGE tensor [H,W,C] float(0..1) -> np.float32 HWC (0..1)
    """
    img = img_t.clamp(0, 1).cpu().numpy()
    return img.astype(np.float32)

def _np_to_tensor(img_np: np.ndarray) -> torch.Tensor:
    """
    np.float32 HWC (0..1) -> ComfyUI IMAGE tensor [H,W,C] float(0..1)
    """
    img = np.clip(img_np, 0.0, 1.0).astype(np.float32)
    return torch.from_numpy(img)

def _soft_light(B: np.ndarray, O_gray: np.ndarray) -> np.ndarray:
    """
    Soft Light blend (Photoshop-like approximation).
    B: HWC in 0..1, O_gray: HW in 0..1
    """
    O = O_gray[..., None]
    return (1.0 - 2.0 * O) * (B ** 2) + 2.0 * O * B

def _gaussian_blur_gray(gray01: np.ndarray, radius: float) -> np.ndarray:
    img = Image.fromarray((np.clip(gray01, 0, 1) * 255.0 + 0.5).astype(np.uint8), "L")
    img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(img).astype(np.float32) / 255.0

def _skin_mask_from_rgb(rgb01: np.ndarray) -> np.ndarray:
    """
    Heuristic skin mask using YCrCb + HSV thresholds.
    Input: np.float32 HWC (0..1). Returns float mask in 0..1 (HW).
    """
    rgb8 = (np.clip(rgb01, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    pil = Image.fromarray(rgb8, "RGB")

    # YCrCb thresholds
    ycbcr = pil.convert("YCbCr")
    y, cb, cr = [np.asarray(c) for c in ycbcr.split()]
    m_ycrcb = (cr > 132) & (cr < 179) & (cb > 77) & (cb < 135)

    # HSV thresholds
    hsv = pil.convert("HSV")
    h, s, v = [np.asarray(c) for c in hsv.split()]
    m_hsv = ((h < 30) | (h > 330)) & (s > 25) & (s < 220) & (v > 40) & (v < 250)

    m = (m_ycrcb & m_hsv).astype(np.uint8) * 255
    m = Image.fromarray(m, "L")
    # Thicken a touch + denoise + feather
    m = m.filter(ImageFilter.MaxFilter(3))
    m = m.filter(ImageFilter.MedianFilter(5))
    m = m.filter(ImageFilter.GaussianBlur(radius=2.0))
    return np.asarray(m).astype(np.float32) / 255.0

def _bandpass_noise(h: int, w: int, r1: float = 0.65, r2: float = 1.9, seed: int | None = None) -> np.ndarray:
    """
    Procedural pores noise: difference of Gaussians to keep only pore-scale detail.
    Returns HW in 0..1.
    """
    rng = np.random.default_rng(seed)
    n = rng.random((h, w)).astype(np.float32)
    b1 = _gaussian_blur_gray(n, r1)
    b2 = _gaussian_blur_gray(n, r2)
    bp = b1 - b2
    # Normalize to 0..1 centered near 0.5
    bp = (bp - bp.min()) / (bp.max() - bp.min() + 1e-6)
    return bp

def _make_microtexture_tile(size: int = 1024, seed: int = 42, contrast: float = 0.35) -> Image.Image:
    """
    Neutral gray high-frequency tile for soft overlay. Returns PIL 'L'.
    """
    rng = np.random.default_rng(seed)
    base = (rng.random((size, size)) * 255).astype(np.uint8)
    img = Image.fromarray(base, mode="L")

    blur1 = img.filter(ImageFilter.GaussianBlur(0.8))
    blur2 = img.filter(ImageFilter.GaussianBlur(1.6))
    blur3 = img.filter(ImageFilter.GaussianBlur(3.2))

    def scale_uint8(imgPIL: Image.Image, factor: float) -> Image.Image:
        return imgPIL.point(lambda p: int(max(0, min(255, round(p * factor)))))

    import PIL.ImageChops as IC
    hp1 = IC.subtract(img, blur1, scale=1.0, offset=128)
    hp2 = IC.subtract(img, blur2, scale=1.0, offset=128)
    hp3 = IC.subtract(img, blur3, scale=1.0, offset=128)
    hp = IC.add_modulo(scale_uint8(hp1, 0.6), scale_uint8(hp2, 0.3))
    hp = IC.add_modulo(hp, scale_uint8(hp3, 0.1))

    arr = np.array(hp).astype(np.float32) / 255.0
    arr = 0.5 + (arr - 0.5) * contrast
    arr = np.clip(arr, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8), mode="L")

def _tile_to_size(texL: Image.Image, w: int, h: int) -> np.ndarray:
    tw, th = texL.size
    nx = (w + tw - 1) // tw
    ny = (h + th - 1) // th
    canvas = Image.new("L", (nx * tw, ny * th), 128)
    for yy in range(ny):
        for xx in range(nx):
            canvas.paste(texL, (xx * tw, yy * th))
    canvas = canvas.crop((0, 0, w, h))
    return np.asarray(canvas).astype(np.float32) / 255.0

# ---------------------------
# Core effect (simple API)
# ---------------------------

def apply_real_skin_simple(rgb01: np.ndarray,
                           strength: float = 0.09,
                           multiply: float = 0.04,
                           seed: int | None = None,
                           mask01: np.ndarray | None = None) -> np.ndarray:
    """
    Minimal version exposing only Strength/Multiply.
    - strength: Soft Light opacity of microtexture (0..~0.3)
    - multiply: small Multiply pass (0..~0.15)
    """
    h, w, _ = rgb01.shape
    if mask01 is None:
        mask01 = _skin_mask_from_rgb(rgb01)
    mask3 = mask01[..., None]

    # Microtexture overlay
    tex = _make_microtexture_tile(seed=seed if seed is not None else 42)
    T = _tile_to_size(tex, w, h)  # 0..1 gray
    soft = _soft_light(rgb01, T)

    out = rgb01 * (1 - mask3) + soft * mask3  # Soft Light (full inside mask), then scale via strength
    out = rgb01 * (1 - strength) + out * strength  # global mix of effect via 'strength'

    if multiply > 0.0:
        mul = rgb01 * T[..., None]
        out = out * (1 - multiply) + mul * multiply

    return np.clip(out, 0.0, 1.0)

# ---------------------------
# ComfyUI Node: Simple
# ---------------------------

class RealSkinPostSimple:
    """
    Post-process node adding realistic microtexture over detected skin.
    Only two knobs: Strength (Soft Light) and Multiply.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 0.09, "min": 0.0, "max": 0.5, "step": 0.005}),
                "multiply": ("FLOAT", {"default": 0.04, "min": 0.0, "max": 0.3, "step": 0.005}),
            },
            "optional": {
                "mask": ("MASK",),          # If given, effect applies only where mask>0 (otherwise auto skin mask)
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31-1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "postprocess/skin"

    def apply(self, image, strength, multiply, mask=None, seed=-1):
        """
        image: torch tensor [B,H,W,C] 0..1
        mask:  torch tensor [B,H,W]  0..1 (optional)
        """
        if seed is not None and seed >= 0:
            np.random.seed(seed)
            random.seed(seed)

        imgs_out = []
        B = image.shape[0]
        for i in range(B):
            img_np = _tensor_to_np(image[i])
            m_np = None
            if mask is not None:
                m_np = mask[i].clamp(0, 1).cpu().numpy().astype(np.float32)

            out_np = apply_real_skin_simple(
                img_np, strength=float(strength), multiply=float(multiply),
                seed=None if seed < 0 else int(seed), mask01=m_np
            )
            imgs_out.append(_np_to_tensor(out_np))

        out_tensor = torch.stack(imgs_out, dim=0)
        return (out_tensor,)

# ---------------------------
# Optional: Advanced node (extra controls)
# ---------------------------

def _luminance(rgb01: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb01[..., 0] + 0.7152 * rgb01[..., 1] + 0.0722 * rgb01[..., 2]

def apply_real_skin_advanced(rgb01: np.ndarray,
                             strength: float, multiply: float,
                             pores_strength: float, pores_min_radius: float, pores_max_radius: float,
                             highlight_breakup: float, color_var: float,
                             seed: int | None, mask01: np.ndarray | None) -> np.ndarray:
    h, w, _ = rgb01.shape
    if mask01 is None:
        mask01 = _skin_mask_from_rgb(rgb01)
    mask3 = mask01[..., None]

    # 1) Tile microtexture
    tex = _make_microtexture_tile(seed=seed if seed is not None else 42)
    T = _tile_to_size(tex, w, h)
    soft = _soft_light(rgb01, T)
    out = rgb01 * (1 - mask3) + soft * mask3
    out = rgb01 * (1 - strength) + out * strength

    if multiply > 0.0:
        mul = rgb01 * T[..., None]
        out = out * (1 - multiply) + mul * multiply

    # 2) Band-pass pores (procedural)
    if pores_strength > 0.0:
        pores = _bandpass_noise(h=h, w=w, r1=pores_min_radius, r2=pores_max_radius, seed=seed)
        out = out * (1 - mask3 * pores_strength) + _soft_light(out, pores) * (mask3 * pores_strength)

    # 3) Highlight breakup (only in bright areas)
    if highlight_breakup > 0.0:
        L = _luminance(out)
        # Use a smoothstep to target highlights ~0.62..0.92
        t = np.clip((L - 0.62) / (0.92 - 0.62 + 1e-6), 0, 1)
        t = t * t * (3 - 2 * t)
        rng = np.random.default_rng(seed)
        sparkle = rng.random((h, w)).astype(np.float32)
        br = 1.0 - t * highlight_breakup * (0.6 + 0.4 * sparkle)
        out = out * (1 - mask3) + (out * br[..., None]) * mask3

    # 4) Low-freq color variance
    if color_var > 0.0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0, 1, (h, w, 3)).astype(np.float32)
        # Blur per-channel for low frequency
        for c in range(3):
            ch = _gaussian_blur_gray(noise[..., c] * 0.5 + 0.5, 6.0)
            noise[..., c] = (ch - 0.5) * 2.0
        out = np.clip(out + noise * color_var * mask3, 0, 1)

    return np.clip(out, 0.0, 1.0)

class RealSkinPostAdvanced:
    """
    Advanced version with extra controls for pore scale/strength and highlight breakup.
    Strength/Multiply are still the primary knobs; others default to sensible values.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 0.6, "step": 0.005}),
                "multiply": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.3, "step": 0.005}),
                "pores_strength": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 0.5, "step": 0.005}),
                "pores_min_radius": ("FLOAT", {"default": 0.65, "min": 0.2, "max": 5.0, "step": 0.05}),
                "pores_max_radius": ("FLOAT", {"default": 1.90, "min": 0.2, "max": 8.0, "step": 0.05}),
                "highlight_breakup": ("FLOAT", {"default": 0.16, "min": 0.0, "max": 0.5, "step": 0.005}),
                "color_var": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.1, "step": 0.001}),
            },
            "optional": {
                "mask": ("MASK",),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**31-1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "postprocess/skin"

    def apply(self, image, strength, multiply, pores_strength, pores_min_radius, pores_max_radius,
              highlight_breakup, color_var, mask=None, seed=-1):
        if seed is not None and seed >= 0:
            np.random.seed(seed)
            random.seed(seed)

        imgs_out = []
        B = image.shape[0]
        for i in range(B):
            img_np = _tensor_to_np(image[i])
            m_np = None
            if mask is not None:
                m_np = mask[i].clamp(0, 1).cpu().numpy().astype(np.float32)

            out_np = apply_real_skin_advanced(
                img_np,
                strength=float(strength),
                multiply=float(multiply),
                pores_strength=float(pores_strength),
                pores_min_radius=float(pores_min_radius),
                pores_max_radius=float(pores_max_radius),
                highlight_breakup=float(highlight_breakup),
                color_var=float(color_var),
                seed=None if seed < 0 else int(seed),
                mask01=m_np,
            )
            imgs_out.append(_np_to_tensor(out_np))

        out_tensor = torch.stack(imgs_out, dim=0)
        return (out_tensor,)

# Required by ComfyUI to discover nodes
NODE_CLASS_MAPPINGS = {
    "RealSkinPostSimple": RealSkinPostSimple,
    "RealSkinPostAdvanced": RealSkinPostAdvanced,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "RealSkinPostSimple": "Real Skin Post (Simple)",
    "RealSkinPostAdvanced": "Real Skin Post (Advanced)",
}
