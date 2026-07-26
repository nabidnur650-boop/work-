# Protocol: cap-corrected post-outcome cross-fitting

## Design boundary

The frozen Perception Test result and all six newly generated single-scale
candidate files existed before this correction. Six descriptive candidate
mAP values had also appeared in an interrupted analyzer's stdout. This record
is therefore post-outcome and cannot establish a new confirmatory or zero-shot
claim.

The interruption occurred before any class router was assembled or scored.
No router prediction or report existed. The correction enforces a constraint
that the original protocol already stated: every method is limited to 50
segments per class/video and 200 segments per video.

## Candidate construction

For CLAP and the unchanged five-seed QCR ensemble, candidates use 0.5, 1, 2,
or 4 second windows, plus the frozen four-scale pool. Prompts, embeddings,
model checkpoints, sigmoid temperature, Gaussian soft-NMS, minimum score, and
the 50-segment class/video cap are unchanged.

The builder retains the per-class pools *before* the 200-segment cross-class
cap. As an implementation audit, applying the ordinary 200-segment cap back
to each pool must reproduce every one of the ten already frozen capped
candidate files row for row under canonical JSON serialization.

## Cross-fitting and final cap

Video fold is the first eight bytes of SHA-256(video_id), interpreted as an
unsigned big-endian integer, modulo five. For held-out fold `k` and class `c`,
the candidate maximizing official mean AP over tIoU 0.1--0.5 on the other four
folds is selected; ties are resolved lexicographically.

For a held-out video, the selected per-class rows are merged, sorted by the
unchanged score/label/time key, and truncated once to 200 rows. The same
procedure is run for the ten-candidate router and a five-candidate CLAP-only
ablation.

## End points

The exploratory endpoint is official class-macro mAP for the cap-corrected
router versus frozen multiscale CLAP. Paired video bootstrap uncertainty uses
10,000 fixed-seed resamples. Diagnostic criteria require a positive official
delta, a positive paired-video lower endpoint, improvement at four of five
tIoU thresholds, no class delta below -0.02 AP, exact cap reconstruction for
all ten candidates, complete held-out assignment, and integrity.

These criteria diagnose whether low-capacity in-domain routing repairs the
failure. They do not replace the original six frozen transfer gates.
