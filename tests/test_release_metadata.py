from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).parents[1]
HF_DIR = ROOT / "release" / "huggingface"
HF_REPOSITORY = "slicedonions/self-driving-rl-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def test_mit_license_covers_root_and_model_package() -> None:
    root_license = (ROOT / "LICENSE").read_text(encoding="utf-8")
    model_license = (HF_DIR / "LICENSE").read_text(encoding="utf-8")
    package = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))[
        "project"
    ]

    assert root_license == model_license
    assert root_license.startswith("MIT License\n")
    assert "Copyright (c) 2026 Shuvra Basak" in root_license
    assert package["license"] == "MIT"
    assert package["license-files"] == ["LICENSE"]
    assert "License :: OSI Approved :: MIT License" in package["classifiers"]


def test_model_card_and_manifest_have_approved_release_metadata() -> None:
    model_card = (HF_DIR / "README.md").read_text(encoding="utf-8")
    manifest = json.loads((HF_DIR / "artifact-manifest.json").read_text(encoding="utf-8"))

    assert "license: mit" in model_card
    assert HF_REPOSITORY in model_card
    assert manifest["publication_status"] == "pending_user_approval"
    assert manifest["license_status"] == "approved"
    assert manifest["code_license"] == manifest["model_license"] == "MIT"
    assert manifest["hugging_face_repository"] == HF_REPOSITORY

    for artifact in manifest["artifacts"]:
        path = HF_DIR / artifact["path"]
        assert path.stat().st_size == artifact["size_bytes"]
        assert _sha256(path) == artifact["sha256"]
        assert artifact["public_download_url"].startswith(
            f"https://huggingface.co/{HF_REPOSITORY}/resolve/main/"
        )


def test_release_documents_have_no_unresolved_url_or_license_placeholder() -> None:
    paths = [
        ROOT / "README.md",
        ROOT / "docs" / "release-artifacts.md",
        ROOT / "release" / "reddit-post.md",
        ROOT / "release" / "v1.0.0-notes.md",
        HF_DIR / "README.md",
    ]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in paths)

    assert "<PENDING" not in combined
    assert "PENDING_" not in combined
    assert "license is still pending" not in combined.casefold()
    assert HF_REPOSITORY in combined
