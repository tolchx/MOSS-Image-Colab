"""Tests del feather/compositing. La garantía importante: fuera del soporte, píxeles intactos."""
import numpy as np
import pytest
from PIL import Image, ImageFilter

from mosskit import masks as MK


def mask_with_blob(size=(256, 256), box=(100, 100, 150, 150)) -> np.ndarray:
    m = np.zeros(size[::-1], dtype=bool)
    x0, y0, x1, y1 = box
    m[y0:y1, x0:x1] = True
    return m


def test_feather_radius_mapping():
    assert MK.feather_radius(1024, 1024, 0.15) == 15
    assert MK.feather_radius(2048, 1024, 0.15) == 15  # usa el lado menor
    assert MK.feather_radius(1024, 1024, 0.0) == 0
    for bad in (-0.1, 1.1, "0.2"):
        with pytest.raises(ValueError):
            MK.feather_radius(1024, 1024, bad)


def test_feathered_alpha_is_one_inside_and_zero_far_outside():
    mask = mask_with_blob()
    radius = 12
    alpha = MK.feathered_alpha(mask, radius)
    assert np.all(alpha[mask] == 1.0)
    # soporte = máscara dilatada exactamente `radius` px (MaxFilter(3) = 1 px por pasada)
    support = Image.fromarray((mask * 255).astype(np.uint8))
    for _ in range(radius):
        support = support.filter(ImageFilter.MaxFilter(3))
    support = np.asarray(support) > 0
    assert np.all(alpha[~support] == 0.0), "el feather se extendió más allá del radio"
    band = alpha[support & ~mask]
    assert band.max() > 0.0, "la banda de feather quedó vacía"
    # el píxel pegado a la máscara sí tiene que estar mezclado
    assert alpha[95:105, 95:105].max() > 0.0
    assert alpha.max() <= 1.0 and alpha.min() >= 0.0


def test_composite_leaves_every_pixel_outside_support_untouched():
    rng = np.random.default_rng(7)
    source = Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8), "RGB")
    generated = Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8), "RGB")
    mask = mask_with_blob()
    alpha = MK.feathered_alpha(mask, 9)
    out = MK.composite(source, generated, alpha)
    assert MK.outside_error(source, out, alpha) == 0
    inside = np.asarray(out)
    assert np.array_equal(inside[mask], np.asarray(generated)[mask]), "adentro debe mandar la salida"


def test_composite_rejects_size_mismatch():
    a = Image.new("RGB", (64, 64))
    b = Image.new("RGB", (32, 32))
    with pytest.raises(ValueError):
        MK.composite(a, b, np.zeros((64, 64), dtype=np.float32))


def test_binary_mask_prefers_alpha_then_luminance():
    rgba = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    rgba.paste((0, 0, 0, 255), (0, 0, 5, 10))
    m = MK.binary_mask(rgba)
    assert m[:, :5].all() and not m[:, 5:].any()
    grey = Image.new("RGB", (10, 10), (0, 0, 0))
    grey.paste((255, 255, 255), (0, 0, 4, 10))
    m2 = MK.binary_mask(grey)
    assert m2[:, :4].all() and not m2[:, 4:].any()


def test_resize_mask_stays_binary():
    mask = mask_with_blob((128, 128), (10, 10, 40, 40))
    big = MK.resize_mask(mask, (512, 512))
    assert big.dtype == bool
    assert set(np.unique(big)) <= {True, False}
    assert big.sum() > mask.sum()


def test_mask_png_is_greyscale_rgb():
    alpha = MK.feathered_alpha(mask_with_blob(), 4)
    img = MK.mask_png(alpha)
    assert img.mode == "RGB" and img.size == (256, 256)
    assert np.asarray(img)[..., 0].max() == 255
