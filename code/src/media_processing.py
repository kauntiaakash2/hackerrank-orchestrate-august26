"""Deterministic media extraction with persistent, failure-safe caching."""
from __future__ import annotations
import hashlib, json, logging
from pathlib import Path
from typing import Any
from PIL import Image

LOG = logging.getLogger(__name__)

class MediaProcessor:
    def __init__(self, dataset_dir: Path, cache_path: Path):
        self.dataset_dir, self.cache_path = dataset_dir, cache_path
        try: self.cache: dict[str, Any] = json.loads(cache_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError): self.cache = {}

    def extract(self, media_type: str, media_id: str, path: str) -> dict[str, Any]:
        if not media_id: return {"text":"", "confidence":1.0, "status":"not_applicable"}
        p=self.dataset_dir/path; digest=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "missing"
        key=f"{media_id}:{digest}"
        if key in self.cache: return self.cache[key]
        result={"text":"", "confidence":0.2, "status":"fallback", "sha256":digest}
        try:
            if media_type=="image":
                with Image.open(p) as im: result.update(width=im.width,height=im.height,format=im.format,status="image_inspected",confidence=0.35)
            elif media_type=="voice": result.update(status="asr_unavailable",confidence=0.15)
        except Exception as exc: LOG.warning("media extraction failed for %s: %s",media_id,exc)
        self.cache[key]=result; self.cache_path.parent.mkdir(parents=True,exist_ok=True); self.cache_path.write_text(json.dumps(self.cache,indent=2,sort_keys=True),encoding="utf-8")
        return result
