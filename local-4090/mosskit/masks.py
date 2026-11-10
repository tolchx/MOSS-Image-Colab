"""Máscaras de inpaint con soporte finito + compositing con garantía por píxel.

Adaptado de src/moss_studio/domain/masks.py del repo MOSS-Image-Colab (Apache-2.0).

Garantía (la que el repo llama AC-09): todo píxel que esté a más de `radius` px de la zona
pintada se copia tal cual del original, byte por byte. Solo la banda de `radius` px hacia
afuera mezcla. Está cubierto por tests/test_masks.py.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter


def feather_radius(width: int, height: int, edge_softness: float) -> int:
    """Mapeo del repo: 1024² con softness 0.15 → 15 px."""
    if not isinstance(edge_softness, (int, float)) or not 0 <= edge_softness <= 1:
        raise ValueError("edge_softness tiene que estar entre 0 y 1")
    return round(0.1 * min(width, height) * edge_softness)


def binary_mask(image: Image.Image) -> np.ndarray:
    """Capa de pincel con transparencia: cualquier trazo opaco cuenta. Sin alfa: blanco = repintar."""
    if "A" in image.getbands():
        alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
        if (alpha < 255).any():
            return alpha > 127
    return np.asarray(image.convert("L"), dtype=np.uint8) > 127


def feathered_alpha(mask: np.ndarray, radius: int) -> np.ndarray:
    """1.0 dentro de la máscara, rampa a 0 en `radius` px hacia afuera, 0 exacto más allá.

    MaxFilter(3) agranda el soporte 1 px por pasada, así que el soporte real es el radio pedido:
    el difuminado nunca se extiende más lejos que `support`.
    """
    selected = mask.astype(np.float32)
    if radius <= 0:
        return selected
    image = Image.fromarray((mask * 255).astype(np.uint8))
    support = image
    remaining = radius
    while remaining > 0:
        support = support.filter(ImageFilter.MaxFilter(3))
        remaining -= 1
    blurred = np.asarray(image.filter(ImageFilter.GaussianBlur(radius / 2)), dtype=np.float32)
    ramp = np.clip(blurred / 127.5, 0.0, 1.0)
    alpha = np.maximum(selected, ramp) * (np.asarray(support) > 0)
    return alpha.astype(np.float32)


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray((mask * 255).astype(np.uint8)).resize(size, Image.Resampling.NEAREST)
    return np.asarray(image) > 127


def composite(source: Image.Image, generated: Image.Image, alpha: np.ndarray) -> Image.Image:
    if source.size != generated.size or alpha.shape != (source.height, source.width):
        raise ValueError("el tamaño de la salida y el de la máscara no coinciden")
    original = np.asarray(source.convert("RGB"), dtype=np.float32)
    produced = np.asarray(generated.convert("RGB"), dtype=np.float32)
    weight = alpha[..., None]
    blended = np.rint(original * (1 - weight) + produced * weight)
    # Donde el soporte es 0 se conservan los bytes originales, no un ida y vuelta por float.
    result = np.where(weight > 0, blended, original).astype(np.uint8)
    return Image.fromarray(result)


def outside_error(source: Image.Image, result: Image.Image, alpha: np.ndarray) -> int:
    """Cuántos píxeles fuera del soporte difieren del original. Tiene que dar 0."""
    outside = alpha <= 0
    a = np.asarray(source.convert("RGB"))[outside]
    b = np.asarray(result.convert("RGB"))[outside]
    return int((a != b).any(axis=1).sum())


def mask_png(alpha: np.ndarray) -> Image.Image:
    """Máscara gris para LoadImageMask (canal rojo). 255 = repintar."""
    return Image.fromarray(np.rint(alpha * 255).astype(np.uint8)).convert("RGB")
