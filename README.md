# MOSS-Image on a single GPU (RTX 4090) — local toolkit + measured findings

This repository is a fork of **[Openclaw-Metis/MOSS-Image-Colab](https://github.com/Openclaw-Metis/MOSS-Image-Colab)**,
a Google Colab studio (FastAPI + Gradio over a headless ComfyUI) for [MOSS-Image](https://github.com/OpenMOSS/MOSS-Image).

The upstream studio is tied to Colab: its bootstrap hard-fails on any GPU that is not an NVIDIA L4
(`l4_preflight`), it pins its own ComfyUI copy, and everything lives on the ephemeral disk.
MOSS-Image has had native ComfyUI support since 0.43/0.44, so a 24 GB desktop card runs it
roughly **10× faster than the L4** with no Colab session at all.

So this fork keeps the reusable parts of the studio as a small, testable Python package
(`local-4090/`) and documents what actually works, measured on an RTX 4090.

The upstream Colab studio is untouched and still here: `src/`, `notebooks/`, `docs/`, `specs/`,
`workflows/`, `configs/`. Its documentation moved to **[README-colab-studio.md](README-colab-studio.md)**
(Traditional Chinese).

---

## What `local-4090/` contains

| Module | What it does | Came from |
|---|---|---|
| `mosskit/layout.py` | Compiles boxes → the JSON layout prompt MOSS reads as text (`abs_centralized`, `yxyx`) | Port of `src/moss_studio/domain/layout.py` |
| `mosskit/masks.py` | Finite-support feather + compositing that keeps untouched pixels byte-identical | Port of `src/moss_studio/domain/masks.py` |
| `mosskit/profiles.py` | Sampling profiles with the parameters already validated on GPU | `configs/profiles/*.json` + ComfyUI templates |
| `mosskit/graph.py` | Builds complete ComfyUI API graphs (t2i, edit, inpaint) with no runtime patching | `workflows/*.json` |
| `mosskit/client.py` | Upload, queue, poll, download | — |
| `tests/` | 17 tests, no GPU needed | — |

Two behaviours differ from upstream on purpose, both discovered by testing:

1. **`caption == text` for text boxes.** If a text box's `caption` differs from its `text`, the model
   renders the caption as ghost type on the poster. The upstream schema allows the difference; in
   practice it leaks. `compile_layout(..., force_caption_equal_text=True)` enforces the safe form and
   puts the typographic description in `summary.caption` instead.
2. **`text_safe` per profile.** The 8-step DMD8 profile is 3× faster but destroys typography, so it is
   flagged and documented instead of being offered as an equal option.

---

## Measured on an RTX 4090 (24 GB)

ComfyUI 0.44.0 (`1332fab`), Python 3.13, torch 2.14.1+cu130, int8 checkpoints.
1 MP unless noted, models already resident.

| Run | Time |
|---|---|
| 25 steps (official template) | **10.2 s** |
| 50 steps | **17.2 s** |
| 1536×864 (16:9) | 12.2 s |
| 8 steps + DMD8 LoRA | 3.4 s |
| Image-edit LoRA, 30 steps | 26–35 s |
| Masked inpaint, 30 steps, denoise 0.65 | ~37 s |
| Cold model load (first run) | ~150 s |

Peak VRAM during generation: **19.2 GB of 24.5 GB**.

For reference, upstream measured the same model at 110–150 s for 50 steps on a Colab L4.

## Text rendering, measured

Tested with 12 generations and a character-by-character visual review.

| Case | Result |
|---|---|
| Short Spanish headline + subline | **10/10 correct**, including `Ñ`, accents and `·` |
| Four-line Spanish paragraph | 1 error in one sample (`Cuatro` → `Cuatra`) |
| 8-step DMD8 profile | **Typography unreadable — unusable for text** |
| 16:9 title card | Correct |

MOSS-Image is bilingual (English/Chinese) by training; Spanish still renders well because the text
encoder is a general 9B model, but budget a proofread on long copy.

## Quickstart

### 1. Pinned ComfyUI, separate from your daily install

```bash
git clone https://github.com/Comfy-Org/ComfyUI.git
cd ComfyUI && git checkout 1332fabfcc507aa7c548cd5152071ed494c207e8
python -m venv ../venv
../venv/Scripts/python -m pip install "torch==2.14.1+cu130" torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu130
../venv/Scripts/python -m pip install -r requirements.txt
```

### 2. Weights (17.2 GB total, from `Comfy-Org/MOSS-Image`)

| File | Size | Folder |
|---|---|---|
| `moss_image_sft_int8_convrot.safetensors` | 5.41 GB | `models/diffusion_models/` |
| `qwen3.5_9b_int8_convrot.safetensors` | 9.75 GB | `models/text_encoders/` |
| `flux2-vae_small.safetensors` | 0.25 GB | `models/vae/` |
| `moss_image_edit_lora_rank_64_bf16.safetensors` | 0.30 GB | `models/loras/` |
| `moss_image_sft_dmd8_lora.safetensors` (optional, 8-step) | 1.19 GB | `models/loras/` |
| `moss_image_color_depth_control.safetensors` (optional) | 0.31 GB | `models/loras/` |

```bash
python main.py --port 8189 --listen 127.0.0.1 --disable-auto-launch
```

### 3. Toolkit

```bash
cd local-4090 && pip install -e ".[dev]" && pytest
python examples/01_poster_layout.py --out out
python examples/02_inpaint.py out/poster_42.png --box 0.30,0.55,0.70,0.92 \
  --prompt "speaker stacks silhouette with cyan halo" --out out
```

The inpaint example prints how many pixels changed outside the feather support. It must be `0`.

### Layout prompt, by hand

```python
from mosskit import layout
prompt = layout.dumps(layout.compile_layout(
    "Minimal black poster, Swiss typography, cyan hairline grid",
    [{"label": "text", "text": "NOCTURNO", "bbox": (0.08, 0.10, 0.92, 0.30)},
     {"label": "text", "text": "Sábado 22 · 21 h", "bbox": (0.08, 0.34, 0.92, 0.42)},
     {"label": "object", "caption": "speaker stacks in silhouette",
      "bbox": (0.30, 0.55, 0.70, 0.92)}],
    1024, 1024))
```

`bbox` is normalised `(x0, y0, x1, y1)`; the compiler converts to pixels relative to the canvas centre.

---

## Pitfalls found the hard way

- **ComfyUI < 0.44 has no `nodes_moss_image.py`.** Don't upgrade your main install in place if it has
  custom nodes — use a separate checkout, as above.
- **`TextEncodeMOSSImageEdit` autogrow inputs are 0-indexed in the API:** it is `images.image_0`,
  not `image_1`. Sending `image_1` fails with `required_input_missing` and a misleading name.
- **Windows batch + SQLite URL:** `--database-url sqlite:///C:\path\comfy.db` (backslashes) hangs
  startup silently after "Setting input directory". Use `sqlite:///C:/path/comfy.db`.
- **4090 VRAM headroom is thin.** 19.2 GB of 24.5 GB means the instance must be closed before running
  another ComfyUI or a game, or you get an OOM instead of a slow run.
- **The edit LoRA is not surgical.** An instruction like "change only the headline text" produced the
  exact new text but reinvented the composition. For a targeted change, regenerate with a layout
  prompt rather than editing.
- **`fast8` is for composition exploration only** — typography comes out unreadable.
- **Layout control via outlines fails on studio/indoor backgrounds** (upstream finding, not re-tested
  here): the estimated silhouette is all white, so depth is the usable control channel.

## Credits and license

- MOSS-Image model and weights: [OpenMOSS](https://github.com/OpenMOSS/MOSS-Image), Apache-2.0.
- Colab studio, layout/mask semantics, sampling profiles: [Openclaw-Metis/MOSS-Image-Colab](https://github.com/Openclaw-Metis/MOSS-Image-Colab), Apache-2.0.
- This fork: same license, see `LICENSE` and `NOTICE`.
