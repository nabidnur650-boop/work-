#!/usr/bin/env python3
"""Refresh the root manifest after accountable publication-metadata edits."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "RELEASE_MANIFEST.json"
MAXIMUM_FILE_BYTES = 100 * 1024 * 1024
MAXIMUM_REPOSITORY_BYTES = 1024 * 1024 * 1024
IGNORED_DIRECTORIES = {"__pycache__", ".pytest_cache", ".ruff_cache", ".venv"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def ignored(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (
        ".git" in relative.parts
        or any(part in IGNORED_DIRECTORIES for part in relative.parts)
        or path.suffix.lower() in {".pyc", ".pyo"}
    )


def verify_accountable_metadata() -> dict[str, Any]:
    license_path = ROOT / "LICENSE"
    citation_path = ROOT / "CITATION.cff"
    if not license_path.is_file() or not citation_path.is_file():
        raise RuntimeError(
            "approved LICENSE and CITATION.cff are required before refresh"
        )
    license_text = license_path.read_text(encoding="utf-8")
    forbidden = ("<COPYRIGHT HOLDER>", "Research package recipient")
    if any(value in license_text for value in forbidden):
        raise RuntimeError("LICENSE still contains a non-accountable holder")

    citation = yaml.safe_load(citation_path.read_text(encoding="utf-8"))
    if not isinstance(citation, dict):
        raise RuntimeError("CITATION.cff is not a mapping")
    required = {
        "cff-version",
        "message",
        "title",
        "authors",
        "version",
        "date-released",
        "repository-code",
    }
    missing = sorted(required - set(citation))
    if missing:
        raise RuntimeError(f"CITATION.cff is missing fields: {missing}")
    authors = citation["authors"]
    if not isinstance(authors, list) or not authors:
        raise RuntimeError("CITATION.cff must contain accountable authors")
    if any(
        not isinstance(author, dict)
        or not (
            {"family-names", "given-names"} <= set(author)
            or bool(author.get("name"))
        )
        for author in authors
    ):
        raise RuntimeError("CITATION.cff contains an incomplete author")
    repository = str(citation["repository-code"])
    if not repository.startswith("https://github.com/"):
        raise RuntimeError("CITATION.cff repository-code must be a GitHub URL")
    return {
        "authors": len(authors),
        "license_sha256": sha256(license_path),
        "citation_sha256": sha256(citation_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--confirm-accountable-metadata",
        action="store_true",
        help="confirm that humans approved the identity and legal metadata",
    )
    args = parser.parse_args()
    if not args.confirm_accountable_metadata:
        parser.error("--confirm-accountable-metadata is required")
    metadata = verify_accountable_metadata()

    files: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == MANIFEST or ignored(path):
            continue
        if path.is_symlink():
            raise RuntimeError(f"symbolic link is not allowed: {path}")
        relative = path.relative_to(ROOT).as_posix()
        if path.suffix.lower() == ".zip":
            raise RuntimeError(f"ZIP must remain outside Git: {relative}")
        size = path.stat().st_size
        if size > MAXIMUM_FILE_BYTES:
            raise RuntimeError(f"file exceeds 100 MiB: {relative}")
        files[relative] = {"bytes": size, "sha256": sha256(path)}
        total_bytes += size
    if total_bytes >= MAXIMUM_REPOSITORY_BYTES:
        raise RuntimeError("repository exceeds the 1-GiB target")

    payload = {
        "schema": "two_study_q1_upgrade_github_manifest_v3",
        "files": files,
        "file_count": len(files),
        "total_bytes": total_bytes,
    }
    MANIFEST.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "root_manifest_refreshed_after_accountable_metadata",
                "files": len(files),
                "total_bytes": total_bytes,
                "metadata": metadata,
                "next": (
                    "git add -A && python verify_release.py --full && "
                    "git diff --cached --check"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
