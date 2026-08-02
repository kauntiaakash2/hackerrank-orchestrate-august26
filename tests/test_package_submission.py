import hashlib
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.package_submission import MANIFEST_NAME, build_archive


def test_archive_is_deterministic_and_manifest_matches(tmp_path: Path) -> None:
    first, second = tmp_path / "first.zip", tmp_path / "second.zip"
    assert build_archive(first) == build_archive(second)
    assert first.read_bytes() == second.read_bytes()

    with zipfile.ZipFile(first) as archive:
        names = archive.namelist()
        assert names[:-1] == sorted(names[:-1])
        assert names[-1] == MANIFEST_NAME
        assert "EVALUATION.md" in names
        assert "cache/media_extractions.json" in names
        assert not any("__pycache__" in name or ".pytest_cache" in name for name in names)
        assert not any(name.endswith((".pyc", ".pyo", ".zip")) for name in names)
        manifest = archive.read(MANIFEST_NAME).decode().splitlines()
        expected = {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in names
            if name != MANIFEST_NAME
        }
        actual = {line.split("  ", 1)[1]: line.split("  ", 1)[0] for line in manifest}
        assert actual == expected


def test_archive_does_not_include_secrets(tmp_path: Path) -> None:
    archive_path = tmp_path / "code.zip"
    build_archive(archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
    assert ".env.example" in names
    assert ".env" not in names
