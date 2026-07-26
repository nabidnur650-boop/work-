# Protocol: cross-fitted class/scale routing after failed zero-shot transfer

## Scientific question

The frozen natural panel showed that a globally fixed multiscale QCR residual
underperformed the matched multiscale CLAP prior.  Aggregate results also
showed strong class heterogeneity: short impulsive events benefited from dense
multiscale proposals, while long or diffuse events often favored four-second
windows.  This follow-up tests whether a class-conditional router can recover
that heterogeneity with video-disjoint supervision.

## Outcome-exposure disclosure

This protocol was written after the four frozen method outcomes were known.
It therefore cannot provide a second confirmatory test on Perception Test
validation.  The lock prevents iterative tuning of the newly generated
single-scale candidates and cross-fitted output, but it does not erase the
earlier outcome exposure.

## Candidate set

For each score source—raw CLAP cosine and the frozen five-seed QCR
ensemble—the candidate set contains 0.5, 1, 2, and 4 second single-scale
predictions plus the original four-scale pool.  Every candidate uses the
unchanged prompts, embeddings, sigmoid transform, Gaussian soft-NMS, score
threshold, and output caps.  No encoder or ranker is retrained.

## Cross-fitting

Video fold is the first eight bytes of SHA-256(video_id), interpreted as an
unsigned integer, modulo five.  For outer fold `k` and class `c`, mean official
AP over tIoU 0.1--0.5 is computed on the other four folds for every candidate.
The highest-AP candidate is selected with lexical tie-breaking and is applied
only to class `c` in fold `k`.  All held-out rows are concatenated and scored
once.  The same procedure is repeated with the CLAP-only candidate subset as
an ablation.

## End points

The primary exploratory endpoint is cross-fitted official mean mAP for the
full router versus the frozen CLAP multiscale comparator.  Secondary outcomes
are threshold and class deltas, the CLAP-only router, fold-wise selection
stability, QCR usage, and paired video-macro-AP uncertainty with 10,000
fixed-seed video bootstrap replicates.

The diagnostic target is met only if the full router improves official mAP,
the paired-video 95% lower endpoint is above zero, at least four of five tIoU
thresholds improve, no class loses more than 0.02 AP, every held-out video is
assigned once, and all source hashes pass.  These are exploratory diagnostic
criteria, not retrospective replacements for the original promotion gates.
