"""Tests del compilador de layout. Sin GPU ni motor: lógica pura."""
import json

import pytest

from mosskit import layout as LY


def item_text(text, bbox, **kw):
    return {"label": "text", "text": text, "bbox": bbox, **kw}


def test_canvas_must_be_multiple_of_16():
    for bad in ((1000, 1024), (1024, 0), (-16, 1024), (1024.0, 1024)):
        with pytest.raises(LY.LayoutError):
            LY.check_canvas(*bad)


def test_bbox_centered_math_full_canvas():
    """Una caja que ocupa todo el lienzo va de -w/2 a +w/2 en x y de -h/2 a +h/2 en y."""
    out = LY.compile_layout("x", [{"label": "object", "caption": "c",
                                   "bbox": (0.0, 0.0, 1.0, 1.0)}], 1024, 1024)
    assert out["items"][0]["bbox"] == [-512, -512, 512, 512]
    assert out["bbox_coord_space"] == "abs_centralized"
    assert out["bbox_format"] == "yxyx"


def test_bbox_order_is_yxyx_not_xyxy():
    """x acotado, y amplio: si el orden fuera xyxy el resultado no daría esto."""
    y0, x0, y1, x1 = LY.to_centralized_yxyx((0.25, 0.10, 0.75, 0.20), 1000, 1000)
    assert (y0, x0, y1, x1) == (-400, -250, -300, 250)


def test_caption_equals_text_for_text_boxes():
    out = LY.compile_layout("brief", [item_text("NOCTURNO", (0.1, 0.1, 0.9, 0.2))], 1024, 1024)
    box = out["items"][0]
    assert box["caption"] == "NOCTURNO"
    assert box["text"] == "NOCTURNO"


def test_caption_is_kept_when_rule_disabled():
    out = LY.compile_layout("brief", [{"label": "text", "text": "A", "caption": "titular grande",
                                       "bbox": (0.1, 0.1, 0.9, 0.2)}], 1024, 1024,
                            force_caption_equal_text=False)
    assert out["items"][0]["caption"] == "titular grande"


def test_rejects_degenerate_and_out_of_range_boxes():
    for bbox in ((0.5, 0.1, 0.5, 0.2), (0.6, 0.1, 0.4, 0.2), (-0.1, 0.0, 0.5, 0.5),
                 (0.0, 0.0, 1.9, 1.0), (0.1, 0.1, 0.2), (0.1, 0.1, 0.2, 0.3, 0.4)):
        with pytest.raises(LY.LayoutError):
            LY.compile_layout("x", [{"label": "object", "caption": "c", "bbox": bbox}], 1024, 1024)


def test_rejects_text_box_without_text_and_object_with_text():
    with pytest.raises(LY.LayoutError):
        LY.compile_layout("x", [{"label": "text", "bbox": (0.1, 0.1, 0.5, 0.2)}], 1024, 1024)
    with pytest.raises(LY.LayoutError):
        LY.compile_layout("x", [{"label": "object", "caption": "c", "text": "no",
                                 "bbox": (0.1, 0.1, 0.5, 0.2)}], 1024, 1024)


def test_item_count_limits():
    one = [{"label": "object", "caption": "c", "bbox": (0.1, 0.1, 0.2, 0.2)}]
    with pytest.raises(LY.LayoutError):
        LY.compile_layout("x", [], 1024, 1024)
    with pytest.raises(LY.LayoutError):
        LY.compile_layout("x", one * 33, 1024, 1024)
    assert len(LY.compile_layout("x", one * 32, 1024, 1024)["items"]) == 32


def test_palette_and_meta_bounds():
    ok = LY.compile_layout("x", [{"label": "object", "caption": "c", "palette": ["#fff"],
                                  "bbox": (0.1, 0.1, 0.2, 0.2)}], 1024, 1024)
    assert ok["items"][0]["palette"] == ["#fff"]
    assert ok["summary"]["meta"] == ["poster"]
    with pytest.raises(LY.LayoutError):
        LY.compile_layout("x", [{"label": "object", "caption": "c", "palette": [],
                                 "bbox": (0.1, 0.1, 0.2, 0.2)}], 1024, 1024)
    with pytest.raises(LY.LayoutError):
        LY.compile_layout("x", [{"label": "object", "caption": "c",
                                 "bbox": (0.1, 0.1, 0.2, 0.2)}], 1024, 1024, meta=["m"] * 9)


def test_serialized_layout_is_valid_json_utf8():
    out = LY.compile_layout("Cartel con ñ y tildes", [item_text("AÑO NUEVO", (0.1, 0.1, 0.9, 0.2))],
                            1024, 1024)
    parsed = json.loads(LY.dumps(out))
    assert parsed["items"][0]["text"] == "AÑO NUEVO"
    assert "\\u" not in LY.dumps(out)  # no escapa el español
