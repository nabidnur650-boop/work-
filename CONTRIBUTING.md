# Contributing

This repository is a frozen research artifact, so evidence changes require
more control than ordinary software changes.

## Suitable contributions

- verifier portability and error-message improvements;
- documentation and accessibility corrections;
- tests that do not alter frozen outcomes;
- clearly separated replications using new namespaces and manifests; and
- security or dependency maintenance.

## Evidence-changing contributions

Do not overwrite frozen protocols, locks, raw records, decisions, figures, or
manuscripts in place. A new experiment must:

1. use a new versioned directory;
2. state whether it is prespecified, confirmatory, replication, or post hoc;
3. retain the prior result and invalidations;
4. provide raw machine-readable records, seeds, gates, and hashes; and
5. update claims only to the extent supported by the new evidence.

## Pull-request checks

Before opening a pull request:

```bash
python -m pip install -r requirements-verify.txt
python verify_release.py --full
git diff --check
```

Explain the scientific claim impact, list changed artifacts, and confirm that
no credentials, private paths, provider-controlled data, audio, or unlicensed
weights were added. The pull-request template captures these declarations.
