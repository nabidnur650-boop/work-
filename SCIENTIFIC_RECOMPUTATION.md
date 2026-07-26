# Independent scientific recomputation

`verify_scientific_results.py` independently reads the retained raw and
machine-readable artifacts instead of trusting only the aggregate publication
reports.

Exactly recomputed:

- both ShiftTitan cell matrices, geometric effects, win/tie/loss counts, and
  task-block bootstrap intervals;
- controlled and source-disjoint audio metrics, gates, seed effects, and
  hierarchical bootstrap intervals;
- the frozen natural per-video contrast and paired-video bootstrap; and
- held-out answer exact/F1 values, source-macro effects, gates, response
  controls, and paired-source bootstrap intervals.

Exhaustively audited:

- every retained natural and router prediction row, including SHA-256,
  schema, finite values, video coverage, labels, and 200-row caps;
- all class-by-threshold metric arithmetic;
- all candidate-pool counts and exact-cap reconstruction receipts; and
- every retained result/receipt hash used by the recomputation.

Boundary:

Perception Test annotations/audio, model weights, large embedding archives,
and regenerable candidate pools are intentionally not redistributed. The
official natural/router global mAP values therefore receive complete retained
artifact and arithmetic verification, but this compact repository does not
misrepresent that as a fresh ground-truth or inference rerun.

Run the complete release audit with:

```bash
python verify_release.py --full
```
