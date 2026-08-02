"""Build the deterministic, source-only ``code.zip`` submission artifact.

The dataset, generated output, and chat transcript are uploaded separately.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path, PurePosixPath
import stat
import zipfile


ROOT = Path(__file__).resolve().parents[1]
FIXED_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
FILES = (
    ".env.example",
    "DATA_AUDIT.md",
    "EVALUATION.md",
    "README.md",
    "main.py",
    "requirements.txt",
)
DIRECTORIES = ("code", "prompts", "cache", "tests", "tools")
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".zip"}
MANIFEST_NAME = "MANIFEST.sha256"


def submission_files(root: Path = ROOT) -> list[Path]:
    """Return the sorted, validated payload paths relative to *root*."""
    candidates = [root / name for name in FILES]
    for directory in DIRECTORIES:
        candidates.extend(path for path in (root / directory).rglob("*") if path.is_file())

    files: list[Path] = []
    for path in candidates:
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ValueError(f"refusing to package symlink: {relative.as_posix()}")
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        if path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example"):
            continue
        files.append(relative)
    missing = [name for name in FILES if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"required package files missing: {', '.join(missing)}")
    return sorted(set(files), key=lambda item: item.as_posix())


def _manifest(root: Path, files: list[Path]) -> bytes:
    lines = []
    for relative in files:
        digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        lines.append(f"{digest}  {relative.as_posix()}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _write_entry(archive: zipfile.ZipFile, name: str, content: bytes) -> None:
    info = zipfile.ZipInfo(name, FIXED_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    archive.writestr(info, content, compresslevel=9)


def build_archive(output: Path, root: Path = ROOT) -> str:
    """Build *output* and return its SHA-256 digest."""
    output = output.resolve()
    files = submission_files(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        with zipfile.ZipFile(temporary, "w") as archive:
            for relative in files:
                _write_entry(archive, relative.as_posix(), (root / relative).read_bytes())
            _write_entry(archive, MANIFEST_NAME, _manifest(root, files))
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(output.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "code.zip")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    digest = build_archive(args.output)
    print(f"wrote {args.output.resolve()}")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
