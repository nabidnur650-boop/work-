# ShiftTitan + EviAudio research artifacts

> **Technical artifact v1.0.0: verified.** The evidence is frozen
> and outcome-faithful. Accountable-author, repository, and license metadata
> remain intentionally pending.

This repository contains two Q1-facing study packages, their rendered
manuscripts, compact scientific records, deterministic manifests, tests, and
an independent numerical verifier. Negative findings are retained as results;
the release does not convert failed promotion gates into positive claims.

## At a glance

| Study | Target venue | Package readiness | Prespecified headline |
|---|---|---:|---|
| [ShiftTitan](studies/shifttitan/README.md) | Information Sciences | **8.71/10** | Both frozen FEV promotion decisions failed; pathwise safety remains supported |
| [EviAudio](studies/eviaudio/README.md) | IEEE/ACM TASLP | **8.95/10** | Controlled ranking passed; fixed natural transfer and downstream utility failed |

The scores rate research-package rigor, not acceptance probability. See the
[declared rubric and empirical boundaries](RATINGS.md).

## Verify first

Python 3.12 is the audited environment.

```bash
python -m pip install -r requirements-verify.txt
python verify_release.py --full
```

The full command checks both bundle manifests, all retained decision
boundaries, repository size, PDFs, public-path sanitization, credential
patterns, compact tests, and independent scientific recomputation. The fast
stdlib-only integrity check is:

```bash
python verify_release.py
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the exact verification levels
and the boundary around provider-controlled data, audio, and model weights.

## Results without spin

### ShiftTitan

| Panel | Origins | Geometric MASE ratio | Skill (95% task-bootstrap interval) | Cells W/T/L | Gates | Decision |
|---|---:|---:|---:|---:|---:|---|
| Original frozen FEV | 83,886 | 0.999804326 | 1.9567 bp [0.1067, 4.4604] | 9/47/4 | 6/7 | **Fail** |
| Task-disjoint fresh FEV | 15,300 | 0.999149434 | 8.5057 bp [-0.6678, 24.0367] | 14/35/11 | 5/7 | **Fail** |

The labeled post-hoc two-panel synthesis estimates
5.2317 bp (95% interval
[0.3137,
13.2410]);
the family-equal sensitivity estimates
3.1385 bp (95% interval
[0.2182,
6.6096]).
Neither replaces the frozen failures. The supported claim is tight harm
control with selective deployment and a small average effect—not broad
forecasting superiority.

[Main PDF](studies/shifttitan/rendered/main.pdf) ·
[Supplement](studies/shifttitan/rendered/supplement.pdf) ·
[Claim ledger](studies/shifttitan/CLAIM_LEDGER.json)

### EviAudio

| Stage | Scope | Baseline → evaluated system | Gates | Decision |
|---|---:|---:|---:|---|
| Controlled exact onset | 397 examples | AP 0.6090 → 0.8425 | 7/7 | **Pass** |
| Source-disjoint external | 219 examples | AP 0.5408 → 0.7020 | 7/8 | **Fail** |
| Frozen natural transfer | 5,359 videos / 35,625 events | mAP 0.067576 → 0.052603 | 2/6 | **Fail** |
| Post-outcome cap-corrected diagnosis | all natural videos | CLAP-only 0.093921; full 0.096995 | diagnostic | **Not fresh evidence** |
| Held-out answer bridge | 6,576 rows | exact 0.4780 baseline vs. 0.4586 learned | 5/8 | **Fail** |

The positive cap-corrected router is explicitly post-outcome and contributes
only 0.003074 mAP beyond the CLAP-only
router. The defensible conclusion is controlled evidence-ranking confirmation,
failed fixed natural transfer, and a temporal-scale diagnosis—not a
generalized zero-shot success.

[Main PDF](studies/eviaudio/rendered/main.pdf) ·
[Supplement](studies/eviaudio/rendered/supplement.pdf) ·
[Claim ledger](studies/eviaudio/CLAIM_LEDGER.json)

## Repository guide

| Resource | Purpose |
|---|---|
| [Final readiness audit](FINAL_Q1_READINESS_AUDIT.md) | Complete technical and scientific evaluation |
| [Independent recomputation](SCIENTIFIC_RECOMPUTATION.md) | Exact endpoints, exhaustive audits, and stated boundary |
| [Release status](FINAL_RELEASE_STATUS.json) | Machine-readable decisions, test counts, and archive hashes |
| [Release manifest](RELEASE_MANIFEST.json) | SHA-256 and byte count for every tracked artifact except itself |
| [Third-party assets](THIRD_PARTY_ASSETS.md) | Excluded assets and acquisition boundary |
| [Contributing](CONTRIBUTING.md) | Frozen-evidence and change-control rules |
| [Security](SECURITY.md) | Private reporting guidance |
| [Citation status](CITATION.md) | Why citation metadata is not fabricated |

## Release boundary

The deterministic journal-submission ZIP files remain beside the source
projects and are intentionally excluded from Git. This repository contains no
provider audio, benchmark datasets, pretrained weights, or regenerable
candidate pools.

The repository is initialized on `main` and fully staged, but it has no
fabricated commit identity, remote, or push. Before a public release, complete
[the accountable metadata checklist](PUBLICATION_METADATA_REQUIRED.md), approve
[the license status](LICENSE_STATUS.md), and follow
[the upload steps](GITHUB_UPLOAD_STEPS.md).
