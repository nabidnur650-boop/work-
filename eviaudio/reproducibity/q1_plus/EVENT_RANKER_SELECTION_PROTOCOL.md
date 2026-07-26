# Exact-onset event-ranker development selection

**Frozen on 2026-07-22 before any learned ranker was evaluated on the 398-example validation panel.** Frozen CLAP-only validation scores may be computed during embedding construction because CLAP is the declared baseline and has no fitted parameters.

## Candidate and isolation

The sole learned architecture is the bounded CLAP-prior residual ranker in `configs/development_event_ranker.json`. Architecture, loss, optimizer, early stopping, seeds `0..4`, development index, validation index, and original +5-point hit@1 / +0.05 evidence-AP gates are fixed. Development, validation, and sealed confirmatory recipes have disjoint foreground source clusters, event classes, background recordings, and background speakers under the recipe audit.

No AUDITA row, answer, question, audio, category score, exact-onset confirmatory recipe, or confirmatory embedding may be used for model fitting, epoch selection, prompt selection, thresholding, or this decision.

## Five-seed rule

Each declared seed is trained once. Its checkpoint is the epoch maximizing validation evidence AP plus 0.25 times validation hit@1, with the frozen patience rule. This use of validation is confined to epoch selection. The primary deployable ranker is the unweighted mean of the five selected checkpoints' chunk scores. No seed may be chosen post hoc as primary.

The CLAP prior scores, labels, recipe identities, source clusters, class identities, chunk bounds, and target bounds must match exactly across seed artifacts. Checkpoint, raw-row, history, configuration, and both embedding-index hashes must verify.

## Development promotion gates

All conditions are required:

1. Mean seed-level hit@1 gain over CLAP is at least 0.05.
2. Mean seed-level evidence-AP gain is at least 0.05.
3. The fixed five-seed ensemble gains at least 0.05 hit@1 and at least 0.05 evidence AP.
4. Fixed-seed 10,000-replicate hierarchical bootstrap 95% intervals for both ensemble gains have lower endpoints above zero. Resampling order is event class, foreground source cluster within class, then recipe within cluster.
5. At least four of five seeds have positive gains on both hit@1 and evidence AP.
6. In every preregistered duration, position, and SNR subgroup, ensemble hit@1 cannot trail CLAP by more than 0.05.
7. All provenance, checksum, split-isolation, finite-score, row-alignment, and metric-reproduction audits pass.

Recall@4, top-chunk IoU, residual magnitude, individual-seed variance, and subgroup AP are secondary. They cannot override a failed primary condition.

## Stop rule

Failure keeps the 397-example exact-onset confirmatory panel sealed and forbids AUDITA answer evaluation with this ranker generation. Passing authorizes exactly one confirmatory embedding/evaluation run with the frozen five-seed ensemble; it does not by itself establish overall Q1 readiness, which still requires the answer, intervention, multi-backbone, systems, provenance, and human-audit gates in `PREREGISTRATION.md`.
