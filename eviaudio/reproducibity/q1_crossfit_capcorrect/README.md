# Cap-corrected cross-fitted class/scale router

This track repairs a structural defect caught before the first router result:
class-wise candidate merging must be followed by the same 200-segment
per-video cap used by the frozen comparator.

The analysis is post-outcome, in-domain, and exploratory. It is neither a
fresh confirmatory evaluation nor a zero-shot result. The original frozen
natural-panel failure remains unchanged.

Run order:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q q1_crossfit_capcorrect/tests
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python q1_crossfit_capcorrect/scripts/freeze_capcorrect.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python q1_crossfit_capcorrect/scripts/build_candidate_pools.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python q1_crossfit_capcorrect/scripts/analyze_capcorrect_router.py
```
