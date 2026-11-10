"""Ejemplo 2: inpaint con garantía por píxel fuera del borde suave.

Pinta un rectángulo sobre la imagen, repinta solo esa zona y compone el resto con los bytes
originales. Al final verifica que fuera del soporte del feather no cambió ni un píxel.

Uso:
    python examples/02_inpaint.py source.png --box 380,600,700,860 --prompt "..." [--softness 0.15]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mosskit import graph, masks
from mosskit.client import Client


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--box", default="0.30,0.55,0.70,0.92",
                    help="x0,y0,x1,y1 normalizados 0..1, o píxeles si algún valor es > 1")
    ap.add_argument("--prompt", default="torres de parlantes en silueta con halo cian")
    ap.add_argument("--softness", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    source = Image.open(args.source).convert("RGB")
    w, h = source.size

    box = [float(v) for v in args.box.split(",")]
    if max(box) > 1.0:
        x0, y0, x1, y1 = (int(round(v)) for v in box)
    else:
        x0, y0, x1, y1 = (int(round(v * s)) for v, s in zip(box, (w, h, w, h)))
    mask = np.zeros((h, w), dtype=bool)
    mask[y0:y1, x0:x1] = True
    if not mask.any():
        raise SystemExit("la caja quedó vacía")

    radius = masks.feather_radius(w, h, args.softness)
    alpha = masks.feathered_alpha(mask, radius)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_path = out_dir / "moss_source.png"
    mask_path = out_dir / "moss_mask.png"
    source.save(src_path)
    masks.mask_png(alpha).save(mask_path)

    client = Client(os.environ.get("MOSS_COMFY", "http://127.0.0.1:8189"))
    client.wait_ready(seconds=600)
    src_name = client.upload(src_path)
    mask_name = client.upload(mask_path)

    g = graph.inpaint(src_name, mask_name, args.prompt, seed=args.seed)
    result = client.run(g, out_dir, out_name="moss_raw.png")
    raw = Image.open(result["paths"][0]).convert("RGB")

    final = masks.composite(source, raw, alpha)
    final_path = out_dir / "moss_inpaint.png"
    final.save(final_path)

    changed_outside = masks.outside_error(source, final, alpha)
    print(f"feather {radius}px · motor {result['seconds']}s")
    print(f"píxeles alterados fuera del soporte: {changed_outside} (tiene que ser 0)")
    print(f"crudo: {result['paths'][0]}")
    print(f"final: {final_path}")
    return 0 if changed_outside == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
