"""mosskit — piezas portadas del repo MOSS-Image-Colab (Apache-2.0) para uso local en ComfyUI.

Qué se portó y por qué:
  layout.py  ← src/moss_studio/domain/layout.py  (canvas ↔ bbox ↔ JSON de layout de MOSS)
  masks.py   ← src/moss_studio/domain/masks.py   (feather de soporte finito + compositing exacto)
  profiles   ← configs/profiles/*.json            (parámetros medidos en GPU)
  graph.py   ← workflows/*.json                   (grafos API, sin parcheo en runtime)

Origen: https://github.com/Openclaw-Metis/MOSS-Image-Colab (Apache-2.0)
"""

__all__ = ["layout", "masks", "graph", "profiles", "client"]
