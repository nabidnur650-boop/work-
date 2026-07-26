# Reproducibility guide

## Audited environment

- Python 3.12
- dependencies pinned in `requirements-verify.txt`
- Linux used for the final audit; verification itself uses portable Python

Create an isolated environment if desired:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-verify.txt
```

## Level 1: integrity and claim boundaries

This stdlib-only command checks release and study manifests, file sizes, PDF
headers, rating arithmetic, scientific decision labels, excluded candidate
pools, public-path sanitization, credential patterns, and repository-facing
metadata:

```bash
python verify_release.py
```

## Level 2: compact tests and independent recomputation

```bash
python verify_release.py --full
```

In addition to Level 1, this runs the compact shipped test suites and
independently processes:

- 99,186 retained forecast-origin rows;
- 616 controlled/external audio examples;
- all 5,359 natural-panel videos;
- all 5,449,000 retained natural/router localization predictions; and
- all 6,576 held-out answer rows.

The verifier recomputes every endpoint supported by retained raw records and
audits the remaining global-mAP arithmetic from frozen class-by-threshold
arrays. It does not claim to rerun unavailable provider inference.

## Level 3: full provider-dependent reconstruction

A full inference rerun additionally requires the third-party datasets, audio,
pretrained weights, and large intermediate embeddings listed in
`THIRD_PARTY_ASSETS.md`. These files are not redistributed. The study folders
retain acquisition instructions, pinned revisions, identifiers, receipts,
hashes, and reconstruction scripts.

Run provider-dependent scripts only in a clean scientific workspace:
immutable analyzers intentionally refuse to overwrite frozen reports.

## Integrity model

- `RELEASE_MANIFEST.json` covers every repository artifact except itself.
- Each study has an independent `SUBMISSION_MANIFEST.json`.
- Publicly normalized path records preserve both frozen-source and public-copy
  SHA-256 values.
- Deterministic journal ZIP hashes are recorded in
  `FINAL_RELEASE_STATUS.json`; the ZIPs remain outside Git.
