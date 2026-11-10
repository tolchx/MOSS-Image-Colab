# mosskit

Local ComfyUI toolkit for MOSS-Image. See the repository README for the full picture.

```bash
pip install -e ".[dev]"
pytest
python examples/01_poster_layout.py --out out
```

Requires a ComfyUI instance (0.44+) with the MOSS-Image weights loaded, listening on
`http://127.0.0.1:8189` by default. Override with `MOSS_COMFY`.

- `layout.py` — boxes → the JSON layout prompt MOSS-Image reads as text.
- `masks.py` — finite-support feather and compositing that leave outside pixels byte-identical.
- `graph.py` / `profiles.py` — complete API graphs and the sampling parameters measured on GPU.
- `client.py` — upload, queue, poll, download.
