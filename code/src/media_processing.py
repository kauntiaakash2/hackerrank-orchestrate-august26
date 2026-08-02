"""Deterministic, optional local OCR/ASR with versioned content caching."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import hashlib
import importlib
import importlib.util
import json
import logging
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any

from PIL import Image, ImageFilter, ImageOps

LOG = logging.getLogger(__name__)
EXTRACTOR_VERSION = "media-v2"


def _optional_module(name: str) -> Any | None:
    """Import an optional package only when its import metadata is present."""
    if importlib.util.find_spec(name) is None:
        return None
    return importlib.import_module(name)


class MediaProcessor:
    """Extract untrusted media content without requiring optional dependencies."""

    def __init__(self, dataset_dir: Path, cache_path: Path):
        self.dataset_dir, self.cache_path = dataset_dir, cache_path
        self.ocr_timeout = float(os.getenv("ROUTER_OCR_TIMEOUT", "20"))
        self.asr_timeout = float(os.getenv("ROUTER_ASR_TIMEOUT", "120"))
        self.whisper_model = os.getenv("ROUTER_WHISPER_MODEL", "small")
        self.whisper_device = os.getenv("ROUTER_WHISPER_DEVICE", "cpu")
        self.whisper_compute = os.getenv("ROUTER_WHISPER_COMPUTE_TYPE", "int8")
        configured_tesseract = os.getenv("TESSERACT_CMD", "")
        configured_path = Path(configured_tesseract).expanduser() if configured_tesseract else None
        self.tesseract_cmd = (
            str(configured_path) if configured_path and configured_path.is_file()
            else shutil.which(configured_tesseract) if configured_tesseract
            else shutil.which("tesseract") or ""
        )
        self._whisper: Any | None = None
        try:
            value = json.loads(cache_path.read_text(encoding="utf-8"))
            self.cache: dict[str, Any] = value if isinstance(value, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            self.cache = {}

    @property
    def extractor_signature(self) -> str:
        ocr = bool(self.tesseract_cmd and importlib.util.find_spec("pytesseract"))
        asr = importlib.util.find_spec("faster_whisper") is not None
        # Availability is part of the version: an old fallback cannot mask a newly
        # installed extractor, while results remain reusable across media IDs.
        return f"{EXTRACTOR_VERSION}:ocr={int(ocr)}:asr={int(asr)}:model={self.whisper_model}"

    def extract(self, media_type: str, media_id: str, path: str) -> dict[str, Any]:
        if not media_id:
            return {"text": "", "confidence": 1.0, "status": "not_applicable"}
        media_path = self.dataset_dir / path
        digest = hashlib.sha256(media_path.read_bytes()).hexdigest() if media_path.is_file() else "missing"
        key = f"{self.extractor_signature}:{media_type}:{digest}"
        if key in self.cache:
            return self.cache[key]

        result: dict[str, Any] = {
            "text": "", "confidence": 0.0, "status": "media_missing" if digest == "missing" else "fallback",
            "sha256": digest, "extractor_version": self.extractor_signature,
        }
        if digest != "missing":
            try:
                if media_type == "image":
                    result.update(self._extract_image(media_path))
                elif media_type in {"voice", "audio", "voice_note"}:
                    result.update(self._extract_voice(media_path))
                else:
                    result["status"] = "unsupported_media_type"
            except Exception as exc:
                LOG.warning("media extraction failed for %s: %s", media_id, exc)
                result["status"] = "extraction_error"
        self.cache[key] = result
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
        return result

    def _extract_image(self, path: Path) -> dict[str, Any]:
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            details: dict[str, Any] = {"width": image.width, "height": image.height, "format": source.format}
            qr_values = self._detect_qr(image)
            details.update(qr_present=bool(qr_values), qr_values=qr_values)
            if not self.tesseract_cmd or importlib.util.find_spec("pytesseract") is None:
                details.update(text="", confidence=0.15, status="ocr_unavailable")
                return self._image_entities(details)

            pytesseract = _optional_module("pytesseract")
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
            variants = self._image_variants(image)
            candidates = [self._ocr_variant(pytesseract, name, variant) for name, variant in variants]
            best = max(candidates, key=lambda item: (item["confidence"], len(item["text"]), item["variant"]))
            details.update(best, status="ocr_complete" if best["text"] else "ocr_empty")
            return self._image_entities(details)

    @staticmethod
    def _image_variants(image: Image.Image) -> list[tuple[str, Image.Image]]:
        gray = ImageOps.grayscale(image)
        contrast = ImageOps.autocontrast(gray).filter(ImageFilter.SHARPEN)
        threshold = contrast.point(lambda pixel: 255 if pixel >= 160 else 0)
        return [("grayscale", gray), ("autocontrast", contrast), ("threshold160", threshold)]

    def _ocr_variant(self, pytesseract: Any, name: str, image: Image.Image) -> dict[str, Any]:
        output = pytesseract.image_to_data(
            image, lang=os.getenv("ROUTER_OCR_LANGUAGES", "eng"),
            config="--oem 3 --psm 6", output_type=pytesseract.Output.DICT,
            timeout=self.ocr_timeout,
        )
        words, confidences = [], []
        for word, raw_confidence in zip(output.get("text", []), output.get("conf", [])):
            word = " ".join(str(word).split())
            try:
                confidence = float(raw_confidence)
            except (TypeError, ValueError):
                confidence = -1
            if word:
                words.append(word)
                if confidence >= 0:
                    confidences.append(confidence)
        return {"text": " ".join(words), "confidence": round(sum(confidences) / (100 * len(confidences)), 3) if confidences else 0.0, "variant": name}

    @staticmethod
    def _detect_qr(image: Image.Image) -> list[str]:
        if importlib.util.find_spec("cv2") is None or importlib.util.find_spec("numpy") is None:
            return []
        cv2, numpy = _optional_module("cv2"), _optional_module("numpy")
        detector = cv2.QRCodeDetector()
        data, points, _ = detector.detectAndDecode(numpy.asarray(image))
        return [data] if points is not None and data else []

    @staticmethod
    def _image_entities(result: dict[str, Any]) -> dict[str, Any]:
        text = result.get("text", "")
        result["dates"] = sorted(set(re.findall(r"\b(?:\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|\d{1,2}\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)[a-z]*\s*\d{0,4})\b", text, re.I)))
        result["payment_terms"] = sorted(set(match.lower() for match in re.findall(r"\b(?:pay(?:ment)?|invoice|amount due|due today|overdue|upi|cash|₹|rs\.?|usd|inr|scan\s+(?:the\s+)?qr)\b", text, re.I)))
        # Capitalized multi-letter tokens are useful deterministic brand candidates.
        result["brands"] = sorted(set(re.findall(r"\b(?:[A-Z][A-Za-z0-9&.-]{2,}|[A-Z]{2,})\b", text)))[:20]
        return result

    def _extract_voice(self, path: Path) -> dict[str, Any]:
        if importlib.util.find_spec("faster_whisper") is None:
            return {"text": "", "confidence": 0.1, "status": "asr_unavailable", "language": "und"}

        def transcribe() -> tuple[list[Any], Any]:
            if self._whisper is None:
                faster_whisper = _optional_module("faster_whisper")
                self._whisper = faster_whisper.WhisperModel(
                    self.whisper_model, device=self.whisper_device, compute_type=self.whisper_compute,
                )
            segments, info = self._whisper.transcribe(
                str(path), beam_size=1, best_of=1, temperature=0.0,
                condition_on_previous_text=False, vad_filter=False, word_timestamps=False,
            )
            return list(segments), info

        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(transcribe)
        try:
            segments, info = future.result(timeout=self.asr_timeout)
        except FutureTimeout:
            future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            return {"text": "", "confidence": 0.0, "status": "asr_timeout", "language": "und"}
        pool.shutdown(wait=True)
        text = " ".join(" ".join(str(segment.text).split()) for segment in segments).strip()
        probabilities = [max(0.0, min(1.0, math.exp(float(segment.avg_logprob)))) for segment in segments]
        language_probability = float(getattr(info, "language_probability", 0.0) or 0.0)
        acoustic = sum(probabilities) / len(probabilities) if probabilities else 0.0
        confidence = acoustic * language_probability if language_probability else acoustic
        return {"text": text, "confidence": round(confidence, 3), "status": "asr_complete" if text else "asr_empty", "language": getattr(info, "language", "und") or "und", "language_confidence": round(language_probability, 3)}
