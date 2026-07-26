# Reproduction commands

Compact verification:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q   -p no:cacheprovider reproducibility/q1_crossfit_capcorrect/tests
```

After restoring provider-controlled audio, weights, and precompute archives,
the scientific sequence is:

```bash
python q1_plus/scripts/evaluate_event_ranker_confirmatory.py
python q1_90/scripts/evaluate_external_replication.py
python q1_top_tier/scripts/evaluate_perception_test.py
python q1_crossfit_capcorrect/scripts/build_candidate_pools.py
python q1_crossfit_capcorrect/scripts/analyze_capcorrect_router.py
python q1_crossfit_capcorrect/scripts/build_upgrade_publication.py
python q1_crossfit_capcorrect/scripts/build_upgrade_submission.py
```

Immutable analyzers refuse to overwrite existing reports; use a clean
scientific workspace for a full rerun.
