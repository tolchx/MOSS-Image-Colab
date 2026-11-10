"""Ejemplo 1: póster con texto en español y cajas de layout, 25 pasos.

Uso:
    python examples/01_poster_layout.py [--out DIR] [--seed N] [--size 1024x1024]

Necesita el motor arriba:  ./run_comfy.sh   (por defecto en http://127.0.0.1:8189)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mosskit import graph, layout
from mosskit.client import Client

CAPTION = (
    "Cartel minimalista de música electrónica en un puerto: fondo negro, retícula fina cian, "
    "tipografía suiza condensada, halo cian detrás de las torres de parlantes."
)

ITEMS = [
    {"label": "text", "text": "NOCTURNO", "bbox": (0.08, 0.10, 0.92, 0.30)},
    {"label": "text", "text": "Música electrónica · Puerto · Sábado 22",
     "bbox": (0.08, 0.34, 0.92, 0.42)},
    {"label": "object", "caption": "torres de parlantes en silueta con halo cian",
     "bbox": (0.30, 0.55, 0.70, 0.92)},
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--size", default="1024x1024")
    ap.add_argument("--profile", default="quality25", choices=["quality25", "quality50", "fast8"])
    args = ap.parse_args()

    w, h = (int(v) for v in args.size.lower().split("x"))
    prompt = layout.dumps(layout.compile_layout(CAPTION, ITEMS, w, h))

    client = Client(os.environ.get("MOSS_COMFY", "http://127.0.0.1:8189"))
    client.wait_ready(seconds=600)
    g = graph.t2i(prompt, args.profile, width=w, height=h, seed=args.seed)
    result = client.run(g, args.out, out_name=f"poster_{args.seed}.png")
    print(f"listo en {result['seconds']}s → {result['paths'][0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
