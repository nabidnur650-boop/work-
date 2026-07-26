#!/usr/bin/env python3
"""Verify the two upgraded publication bundles and GitHub release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
ROOT_MANIFEST = ROOT / "RELEASE_MANIFEST.json"
MAXIMUM_FILE_BYTES = 100 * 1024 * 1024
MAXIMUM_REPOSITORY_BYTES = 1024 * 1024 * 1024
IGNORED_ARTIFACT_DIRECTORIES = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
}
PUBLIC_TEXT_SUFFIXES = {
    ".cff",
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".tex",
    ".template",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
WEIGHTS = {
    "contribution_originality": 0.15,
    "protocol_validity": 0.20,
    "experimental_depth": 0.15,
    "statistical_rigor": 0.12,
    "reproducibility_auditability": 0.15,
    "manuscript_visual_quality": 0.10,
    "venue_fit": 0.05,
    "generalization_claim_strength": 0.08,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def is_ignored_artifact(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    return (
        any(part in IGNORED_ARTIFACT_DIRECTORIES for part in relative.parts)
        or path.suffix.lower() in {".pyc", ".pyo"}
    )


def verify_publication_hygiene() -> dict[str, int]:
    private_path_patterns = (
        re.compile(r"/" + r"home/" + r"[A-Za-z0-9._-]+/"),
        re.compile(r"/" + r"Users/" + r"[A-Za-z0-9._-]+/"),
        re.compile(
            r"[A-Za-z]:"
            + re.escape("\\")
            + r"Users"
            + re.escape("\\")
            + r"[A-Za-z0-9._-]+"
            + re.escape("\\")
        ),
    )
    credential_patterns = (
        re.compile(
            r"-{5}BEGIN "
            + r"(?:RSA |EC |DSA |OPENSSH )?"
            + r"PRIVATE KEY-{5}"
        ),
        re.compile("github" + r"_pat_[A-Za-z0-9_]{20,}"),
        re.compile("gh" + r"[pousr]_[A-Za-z0-9]{30,}"),
        re.compile("AK" + r"IA[0-9A-Z]{16}"),
        re.compile("sk" + r"-[A-Za-z0-9]{32,}"),
        re.compile("xox" + r"[baprs]-[A-Za-z0-9-]{20,}"),
    )
    text_files = 0
    for path in sorted(ROOT.rglob("*")):
        if (
            not path.is_file()
            or ".git" in path.parts
            or is_ignored_artifact(path, ROOT)
            or path.suffix.lower() not in PUBLIC_TEXT_SUFFIXES
        ):
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        text_files += 1
        if any(pattern.search(content) for pattern in private_path_patterns):
            raise RuntimeError(
                f"machine-local path remains in public file: {path}"
            )
        if any(pattern.search(content) for pattern in credential_patterns):
            raise RuntimeError(f"credential-like material found: {path}")
    return {
        "text_files_scanned": text_files,
        "machine_local_paths_found": 0,
        "credential_patterns_found": 0,
    }


def verify_public_path_sanitization() -> dict[str, int | str]:
    bundle = ROOT / "studies/eviaudio"
    inventory = read_json(bundle / "SCIENTIFIC_ARTIFACT_INVENTORY.json")
    receipt = inventory["public_path_sanitization"]
    records = receipt["files"]
    if receipt["status"] != "machine_local_paths_removed_from_public_copies":
        raise RuntimeError("unexpected public-path sanitization status")
    if receipt["scientific_values_changed"] is not False:
        raise RuntimeError("public-path receipt weakens the scientific boundary")
    if int(receipt["sanitized_file_count"]) != len(records) or not records:
        raise RuntimeError("public-path sanitization count mismatch")
    for relative, record in records.items():
        public_path = bundle / "reproducibility" / relative
        included = inventory["included_files"].get(relative)
        if included is None or not public_path.is_file():
            raise RuntimeError(f"sanitized public file is absent: {relative}")
        expected = (
            int(record["public_bytes"]),
            str(record["public_sha256"]),
            int(record["source_bytes"]),
            str(record["source_sha256"]),
        )
        observed = (
            public_path.stat().st_size,
            sha256(public_path),
            int(included["source_bytes"]),
            str(included["source_sha256"]),
        )
        if observed != expected:
            raise RuntimeError(f"sanitization provenance mismatch: {relative}")
        if included["sha256"] != record["public_sha256"]:
            raise RuntimeError(f"public inventory hash mismatch: {relative}")
        if record["source_sha256"] == record["public_sha256"]:
            raise RuntimeError(f"sanitization did not change declared file: {relative}")
    if not (bundle / "PUBLIC_PATH_SANITIZATION.md").is_file():
        raise RuntimeError("public-path sanitization note is absent")
    return {
        "status": receipt["status"],
        "sanitized_files_verified": len(records),
    }


def verify_repository_interface() -> dict[str, Any]:
    required = {
        ".editorconfig",
        ".gitattributes",
        ".github/ISSUE_TEMPLATE/bug-report.yml",
        ".github/ISSUE_TEMPLATE/config.yml",
        ".github/ISSUE_TEMPLATE/reproducibility.yml",
        ".github/pull_request_template.md",
        ".github/workflows/verify.yml",
        ".gitignore",
        "CHANGELOG.md",
        "CITATION.md",
        "CONTRIBUTING.md",
        "FINAL_Q1_READINESS_AUDIT.md",
        "GITHUB_UPLOAD_STEPS.md",
        "LICENSE.template",
        "LICENSE_STATUS.md",
        "PUBLICATION_METADATA_REQUIRED.md",
        "README.md",
        "RELEASE_VERSION",
        "REPRODUCIBILITY.md",
        "SECURITY.md",
        "requirements-verify.txt",
    }
    missing = sorted(relative for relative in required if not (ROOT / relative).is_file())
    if missing:
        raise RuntimeError(f"repository interface files are absent: {missing}")

    version = (ROOT / "RELEASE_VERSION").read_text(encoding="utf-8").strip()
    status = read_json(ROOT / "FINAL_RELEASE_STATUS.json")
    if version != "1.0.0" or status["release_version"] != version:
        raise RuntimeError("release version mismatch")

    expected_requirements = {
        "numpy==2.4.6",
        "PyYAML==6.0.3",
        "pytest==9.0.3",
        "scipy==1.17.1",
    }
    observed_requirements = {
        line.strip()
        for line in (ROOT / "requirements-verify.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if observed_requirements != expected_requirements:
        raise RuntimeError("verification dependencies are not exactly pinned")

    workflow = (ROOT / ".github/workflows/verify.yml").read_text(
        encoding="utf-8"
    )
    expected_actions = {
        "actions/checkout": "de0fac2e4500dabe0009e67214ff5f5447ce83dd",
        "actions/setup-python": "a309ff8b426b58ec0e2a45f0f869d46889d02405",
    }
    observed_actions = dict(
        re.findall(r"uses:\s+([^@\s]+)@([0-9a-f]{40})", workflow)
    )
    if observed_actions != expected_actions:
        raise RuntimeError("GitHub Actions are not pinned to audited SHAs")
    workflow_markers = (
        "permissions:\n  contents: read",
        "persist-credentials: false",
        "timeout-minutes: 30",
        "python verify_release.py --full",
    )
    if any(marker not in workflow for marker in workflow_markers):
        raise RuntimeError("GitHub Actions hardening marker is absent")

    active_license = ROOT / "LICENSE"
    license_mode = "accountable_license_pending"
    if active_license.exists():
        text = active_license.read_text(encoding="utf-8")
        if (
            "<COPYRIGHT HOLDER>" in text
            or "Research package recipient" in text
        ):
            raise RuntimeError("active LICENSE contains a placeholder holder")
        license_mode = "accountable_license_present"

    markdown_files = sorted(ROOT.glob("*.md")) + [
        ROOT / "studies/shifttitan/README.md",
        ROOT / "studies/eviaudio/README.md",
    ]
    link_pattern = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
    links_checked = 0
    for document in markdown_files:
        content = document.read_text(encoding="utf-8")
        for raw_target in link_pattern.findall(content):
            target = raw_target.strip().strip("<>").split("#", 1)[0]
            if not target or target.startswith(
                ("http://", "https://", "mailto:")
            ):
                continue
            links_checked += 1
            resolved = (document.parent / target).resolve()
            if not resolved.is_relative_to(ROOT.resolve()) or not resolved.exists():
                raise RuntimeError(
                    f"broken or escaping Markdown link: {document}: {target}"
                )
    if links_checked < 15:
        raise RuntimeError("repository navigation link coverage is unexpectedly low")

    return {
        "release_version": version,
        "required_interface_files": len(required),
        "relative_markdown_links_checked": links_checked,
        "immutable_actions_verified": len(expected_actions),
        "license_mode": license_mode,
    }


def verify_bundle(path: Path, schema: str) -> dict[str, Any]:
    manifest_path = path / "SUBMISSION_MANIFEST.json"
    manifest = read_json(manifest_path)
    if manifest["schema"] != schema:
        raise RuntimeError(f"unexpected schema: {path}")
    observed = {}
    for item in sorted(path.rglob("*")):
        if (
            not item.is_file()
            or item == manifest_path
            or is_ignored_artifact(item, path)
        ):
            continue
        relative = item.relative_to(path).as_posix()
        observed[relative] = {
            "bytes": item.stat().st_size,
            "sha256": sha256(item),
        }
        if item.stat().st_size > MAXIMUM_FILE_BYTES:
            raise RuntimeError(f"file exceeds 100 MiB: {relative}")
    if observed != manifest["files"]:
        raise RuntimeError(f"bundle manifest mismatch: {path}")
    if len(observed) != int(manifest["file_count"]):
        raise RuntimeError(f"bundle count mismatch: {path}")
    for name in ("main.pdf", "supplement.pdf"):
        pdf = path / "rendered" / name
        if not pdf.read_bytes().startswith(b"%PDF"):
            raise RuntimeError(f"invalid PDF: {pdf}")
    return manifest


def verify_root() -> int:
    manifest = read_json(ROOT_MANIFEST)
    observed = {}
    total_bytes = 0
    for path in sorted(ROOT.rglob("*")):
        if (
            ".git" in path.parts
            or not path.is_file()
            or path == ROOT_MANIFEST
            or is_ignored_artifact(path, ROOT)
        ):
            continue
        relative = path.relative_to(ROOT).as_posix()
        if path.suffix.lower() == ".zip":
            raise RuntimeError(f"submission ZIP must remain outside Git: {relative}")
        observed[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        total_bytes += path.stat().st_size
        if path.stat().st_size > MAXIMUM_FILE_BYTES:
            raise RuntimeError(f"root file exceeds 100 MiB: {relative}")
    if observed != manifest["files"]:
        raise RuntimeError("root release manifest mismatch")
    if total_bytes != int(manifest["total_bytes"]):
        raise RuntimeError("root byte count mismatch")
    if total_bytes >= MAXIMUM_REPOSITORY_BYTES:
        raise RuntimeError("release exceeds the 1-GiB target")
    return len(observed)


def verify_ratings() -> dict[str, float]:
    ratings = read_json(ROOT / "RATINGS.json")
    if ratings["weights"] != WEIGHTS:
        raise RuntimeError("rating weights changed")
    if abs(sum(WEIGHTS.values()) - 1.0) > 1e-12:
        raise RuntimeError("rating weights do not sum to one")
    result = {}
    for name, study in ratings["studies"].items():
        scores = study["dimensions"]
        if set(scores) != set(WEIGHTS):
            raise RuntimeError(f"rating dimension mismatch: {name}")
        if any(not 0.0 <= float(value) <= 10.0 for value in scores.values()):
            raise RuntimeError(f"rating out of range: {name}")
        observed = round(
            sum(WEIGHTS[key] * float(scores[key]) for key in WEIGHTS), 2
        )
        declared = float(study["overall_research_package_readiness"])
        if observed != declared or declared < 8.5:
            raise RuntimeError(f"rating calculation mismatch: {name}")
        result[name] = declared
    if (
        ratings["q1_acceptance_guaranteed"] is not False
        or ratings["independent_peer_review_completed"] is not False
    ):
        raise RuntimeError("rating boundary was weakened")
    return result


def verify_decisions() -> dict[str, str]:
    status = read_json(ROOT / "FINAL_RELEASE_STATUS.json")
    shift = read_json(
        ROOT
        / "studies/shifttitan/reproducibility/q1_fresh_replication/"
        "publication_upgrade/SHIFT_UPGRADE_ANALYSIS.json"
    )
    audio = read_json(
        ROOT
        / "studies/eviaudio/reproducibility/q1_crossfit_capcorrect/"
        "publication_upgrade/EVIAUDIO_UPGRADE_ANALYSIS.json"
    )
    shift_original = shift["panels"]["fev_mini"]
    shift_fresh = shift["panels"]["task_disjoint_expansion"]
    shift_posthoc = shift["posthoc_stratified_synthesis"]
    controlled = audio["controlled_confirmatory"]
    external = audio["external_exact_onset"]
    natural = audio["frozen_natural_transfer"]
    diagnostic = audio["cap_corrected_diagnosis"]
    downstream = audio["downstream_answer_bridge"]
    expected = status["scientific_status"]
    conditions = (
        shift_original["status"]
        == expected["shifttitan"]["original_frozen_decision"]
        == "fev_panel_promotion_fail",
        shift_fresh["status"]
        == expected["shifttitan"]["task_disjoint_fresh_decision"]
        == "fresh_fev_expansion_promotion_fail",
        shift_posthoc["replaces_frozen_panel_decisions"] is False,
        controlled["status"] == "confirmatory_exact_onset_gate_pass",
        external["status"] == "external_replication_gate_fail",
        natural["status"] == "natural_panel_promotion_fail",
        diagnostic["status"]
        == "cap_corrected_post_outcome_diagnostic_target_met",
        diagnostic["claim_type"] == "post_outcome_cross_fitted_exploratory",
        downstream["status"] == "heldout_answer_gate_fail_audita_sealed",
        status["q1_acceptance_guaranteed"] is False,
    )
    if not all(conditions):
        raise RuntimeError("scientific decision boundary mismatch")
    if any(
        path.parts[-3:-1] == ("candidate_pools", "pools")
        for path in ROOT.rglob("*")
        if path.is_file()
    ):
        raise RuntimeError("regenerable candidate pools were included")
    required_text = {
        ROOT / "studies/shifttitan/main.tex": (
            "not replace either decision",
            "failed",
            "post hoc",
        ),
        ROOT / "studies/eviaudio/main.tex": (
            "fails zero-shot transfer",
            "post-outcome",
            "does not improve held-out answers",
        ),
    }
    for path, phrases in required_text.items():
        text = path.read_text(encoding="utf-8")
        if any(phrase not in text for phrase in phrases):
            raise RuntimeError(f"claim-boundary text missing: {path}")
    return {
        "shifttitan_original": shift_original["status"],
        "shifttitan_fresh": shift_fresh["status"],
        "eviaudio_natural": natural["status"],
        "eviaudio_diagnostic": diagnostic["status"],
    }


def run_tests() -> dict[str, str]:
    suites = {
        "shifttitan": {
            "base": ROOT / "studies/shifttitan/reproducibility",
            "tests": (
                "q1_fresh_replication/tests",
                "q1_top_tier/tests",
            ),
            "sources": (
                "q1_fresh_replication/src",
                "q1_top_tier/src",
                "src",
            ),
        },
        "eviaudio": {
            "base": ROOT / "studies/eviaudio/reproducibility",
            "tests": (
                "q1_crossfit_capcorrect/tests",
                "q1_upgrade/tests",
                "q1_top_tier/tests",
                "q1_90/tests",
            ),
            "sources": (
                "q1_crossfit_capcorrect/src",
                "q1_upgrade/src",
                "q1_top_tier/src",
                "src",
            ),
        },
    }
    results = {}
    for name, suite in suites.items():
        base = suite["base"]
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONPATH"] = os.pathsep.join(
            str(base / relative) for relative in suite["sources"]
        )
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:cacheprovider",
                *(str(base / relative) for relative in suite["tests"]),
            ],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        if process.returncode != 0:
            raise RuntimeError(
                f"{name} compact tests failed:\n"
                f"{process.stdout}\n{process.stderr}"
            )
        results[name] = process.stdout.strip()
    return results


def run_scientific_recomputation() -> dict[str, Any]:
    process = subprocess.run(
        [sys.executable, str(ROOT / "verify_scientific_results.py")],
        check=False,
        capture_output=True,
        text=True,
    )
    if process.returncode != 0:
        raise RuntimeError(
            "scientific recomputation failed:\n"
            f"{process.stdout}\n{process.stderr}"
        )
    result = json.loads(process.stdout)
    if result["status"] != "scientific_results_independently_recomputed":
        raise RuntimeError("unexpected scientific recomputation status")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    shift = verify_bundle(
        ROOT / "studies/shifttitan",
        "shift_information_sciences_q1_upgrade_bundle_v2",
    )
    audio = verify_bundle(
        ROOT / "studies/eviaudio",
        "eviaudio_taslp_q1_upgrade_bundle_v2",
    )
    result = {
        "status": "q1_upgrade_github_release_verified",
        "root_files": verify_root(),
        "bundle_files": {
            "shifttitan": shift["file_count"],
            "eviaudio": audio["file_count"],
        },
        "ratings": verify_ratings(),
        "decisions": verify_decisions(),
        "publication_hygiene": verify_publication_hygiene(),
        "public_path_sanitization": verify_public_path_sanitization(),
        "repository_interface": verify_repository_interface(),
    }
    if args.full:
        result["tests"] = run_tests()
        result["scientific_recomputation"] = run_scientific_recomputation()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
