"""Media tests use tiny generated fixtures and deterministic backend doubles."""
from pathlib import Path
from types import SimpleNamespace
import time

from PIL import Image

from code.src.media_processing import MediaProcessor


def processor(tmp_path: Path) -> MediaProcessor:
    return MediaProcessor(tmp_path, tmp_path / "cache.json")


def test_image_fallback_is_structured(tmp_path, monkeypatch):
    Image.new("RGB", (24, 12), "white").save(tmp_path / "poster.png")
    media = processor(tmp_path)
    monkeypatch.setattr(media, "tesseract_cmd", "")
    monkeypatch.setattr(media, "_detect_qr", lambda image: [])
    result = media.extract("image", "poster", "poster.png")
    assert result["status"] == "ocr_unavailable"
    assert result["text"] == "" and result["confidence"] < .3
    assert result["qr_present"] is False
    assert result["dates"] == result["payment_terms"] == result["brands"] == []


def test_image_entities_include_payment_pressure_brand_date_and_qr():
    result = MediaProcessor._image_entities({
        "text": "ACME Invoice payment due today 24/08/2026 Scan QR",
        "qr_present": True,
    })
    assert "24/08/2026" in result["dates"]
    assert {"invoice", "payment", "due today", "scan qr"} <= set(result["payment_terms"])
    assert "ACME" in result["brands"] and result["qr_present"]


def test_voice_explicitly_falls_back_without_backend(tmp_path, monkeypatch):
    (tmp_path / "note.wav").write_bytes(b"representative-audio-fixture")
    media = processor(tmp_path)
    original = __import__("code.src.media_processing", fromlist=["importlib"]).importlib.util.find_spec
    monkeypatch.setattr(
        "code.src.media_processing.importlib.util.find_spec",
        lambda name: None if name == "faster_whisper" else original(name),
    )
    result = media.extract("voice", "note", "note.wav")
    assert result == {**result, "text": "", "status": "asr_unavailable", "language": "und"}
    assert result["confidence"] < .3


def test_multilingual_voice_transcript_has_language_and_confidence(tmp_path, monkeypatch):
    path = tmp_path / "note.wav"
    path.write_bytes(b"representative-audio-fixture")
    media = processor(tmp_path)
    segment = SimpleNamespace(text="भुगतान आज करना है", avg_logprob=-.1)
    info = SimpleNamespace(language="hi", language_probability=.95)
    media._whisper = SimpleNamespace(transcribe=lambda *args, **kwargs: ([segment], info))
    monkeypatch.setattr("code.src.media_processing.importlib.util.find_spec", lambda name: object())
    result = media._extract_voice(path)
    assert result["text"] == "भुगतान आज करना है"
    assert result["language"] == "hi" and result["language_confidence"] == .95
    assert result["confidence"] > .8


def test_voice_timeout_is_explicit(tmp_path, monkeypatch):
    path = tmp_path / "slow.wav"
    path.write_bytes(b"representative-audio-fixture")
    media = processor(tmp_path)
    media.asr_timeout = .001
    def slow_transcribe(*args, **kwargs):
        time.sleep(.02)
        return [], SimpleNamespace(language="und", language_probability=0)
    media._whisper = SimpleNamespace(transcribe=slow_transcribe)
    monkeypatch.setattr("code.src.media_processing.importlib.util.find_spec", lambda name: object())
    result = media._extract_voice(path)
    assert result["status"] == "asr_timeout" and result["confidence"] == 0


def test_cache_key_reuses_digest_and_changes_with_extractor(tmp_path, monkeypatch):
    Image.new("RGB", (8, 8), "white").save(tmp_path / "same.png")
    media = processor(tmp_path)
    monkeypatch.setattr(media, "tesseract_cmd", "")
    first = media.extract("image", "one", "same.png")
    second = media.extract("image", "two", "same.png")
    assert first == second and len(media.cache) == 1
    assert all("media-v2" in key for key in media.cache)
