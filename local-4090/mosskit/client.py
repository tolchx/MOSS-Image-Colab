"""Cliente mínimo de ComfyUI: subir imagen, encolar, esperar, bajar resultado."""

from __future__ import annotations

import json
import time
import urllib.parse
import uuid
from pathlib import Path

import requests

BASE = "http://127.0.0.1:8189"


class EngineError(RuntimeError):
    pass


class Client:
    def __init__(self, base: str = BASE, timeout: float = 30.0):
        self.base = base.rstrip("/")
        self.timeout = timeout

    # --- infraestructura ---
    def health(self) -> dict:
        r = requests.get(f"{self.base}/system_stats", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def alive(self) -> bool:
        try:
            self.health()
            return True
        except Exception:
            return False

    def wait_ready(self, seconds: float = 300.0) -> dict:
        deadline = time.time() + seconds
        while time.time() < deadline:
            try:
                return self.health()
            except Exception:
                time.sleep(2)
        raise EngineError(f"el motor no respondió en {seconds:.0f}s ({self.base})")

    def upload(self, path: str | Path) -> str:
        path = Path(path)
        with path.open("rb") as fh:
            r = requests.post(f"{self.base}/upload/image",
                              files={"image": (path.name, fh, "image/png")},
                              data={"overwrite": "true", "type": "input"},
                              timeout=120)
        r.raise_for_status()
        info = r.json()
        name = info.get("name") or path.name
        sub = info.get("subfolder") or ""
        return f"{sub}/{name}" if sub else name

    # --- ejecución ---
    def submit(self, graph: dict) -> str:
        client_id = str(uuid.uuid4())
        r = requests.post(f"{self.base}/prompt",
                          json={"prompt": graph, "client_id": client_id}, timeout=60)
        if r.status_code != 200:
            raise EngineError(f"el motor rechazó el grafo: {r.status_code} {r.text[:600]}")
        return r.json()["prompt_id"]

    def wait(self, prompt_id: str, timeout: float = 900.0) -> dict:
        start = time.time()
        delay = 0.4
        while time.time() - start < timeout:
            time.sleep(delay)
            delay = min(delay * 1.15, 3.0)
            h = requests.get(f"{self.base}/history/{prompt_id}", timeout=30).json()
            if prompt_id in h:
                rec = h[prompt_id]
                status = (rec.get("status") or {}).get("status_str")
                if status == "error":
                    raise EngineError(f"error de ejecución: {json.dumps(rec.get('status'))[:800]}")
                files = [img["filename"] for node in rec.get("outputs", {}).values()
                         for img in node.get("images", [])]
                return {"seconds": round(time.time() - start, 1), "files": files, "raw": rec}
            if requests.get(f"{self.base}/queue", timeout=15).json() == {}:
                # la cola vacía sin historia todavía puede ser el arranque del job
                pass
        raise EngineError(f"timeout esperando {prompt_id}")

    def fetch(self, filename: str, dest_dir: str | Path, out_name: str | None = None) -> Path:
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        r = requests.get(f"{self.base}/view",
                         params={"filename": filename, "type": "output"}, timeout=180)
        r.raise_for_status()
        dest = dest_dir / (out_name or Path(filename).name)
        dest.write_bytes(r.content)
        return dest

    def run(self, graph: dict, dest_dir: str | Path, out_name: str | None = None,
            timeout: float = 900.0) -> dict:
        pid = self.submit(graph)
        info = self.wait(pid, timeout=timeout)
        paths = [self.fetch(f, dest_dir, out_name if len(info["files"]) == 1 else None)
                 for f in info["files"]]
        return {"prompt_id": pid, "seconds": info["seconds"], "paths": [str(p) for p in paths]}
