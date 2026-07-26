# Exact-onset confirmatory evaluation

**Frozen on 2026-07-22 after development promotion and before generating any confirmatory embedding or score.**

## Authorization basis

`event_ranker/five_seed_validation_report.json` passed every condition in `EVENT_RANKER_SELECTION_PROTOCOL.md`. The exact-onset confirmatory panel therefore may be opened once. This authorization applies only to the 397 sealed event-needle recipes and cannot authorize AUDITA answer evaluation, prompt tuning, another ranker, or another evidence-panel attempt.

## Fixed evaluation

- Frozen CLAP revision `8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a` generates 4-second chunks at 2-second hops.
- The five development-selected checkpoints for seeds 0–4 are loaded without refitting.
- The primary ranker is the unweighted arithmetic mean of their chunk scores.
- CLAP-only scores from the same embeddings are the paired baseline.
- Every recipe is evaluated once. No threshold, checkpoint, seed, class, SNR, duration, or position may be removed after scoring.
- The locked precompute wrapper accepts no command-line overrides. It verifies every declared foreground, distractor, and background checksum before running the exact sealed recipe path, model revision, 48 kHz sample rate, chunking, and inference batch size.
- The locked evaluator likewise accepts no overrides. It independently verifies archive checksums and schemas, exact chunk bounds and evidence labels, source isolation, model/checkpoint hashes, finite scores, and the residual bound.
- The only authorized commands, in order, are `python q1_plus/scripts/run_confirmatory_event_precompute.py` and `python q1_plus/scripts/evaluate_event_ranker_confirmatory.py` from the project root.
- A completed result is immutable. An operational interruption before a completed precompute receipt or final score artifact may be rerun with the identical locked command; it does not permit any model, data, or setting change.

## Confirmatory evidence gate

All conditions are required:

1. Ensemble hit@1 gain is at least 0.05.
2. Ensemble evidence-AP gain is at least 0.05.
3. Hierarchical 10,000-replicate bootstrap 95% interval lower endpoints for both gains exceed zero; resampling is event class, foreground source cluster, then recipe.
4. At least four of five fixed seeds have positive hit@1 and AP gains.
5. No duration, position, or SNR subgroup loses more than 0.05 hit@1.
6. All recipe, archive, checkpoint, code, protocol, alignment, finite-score, source-isolation, and checksum audits pass.

Recall@4, top-chunk IoU, per-seed variance, residual magnitude, latency, memory, and subgroup AP are secondary. The confirmatory result is immutable whether it passes or fails.

Passing closes only the exact-onset evidence gate in the broader preregistration. Overall Q1 readiness still requires answer-quality, evidence intervention, shuffle/silence, six-category, two-backbone, systems, provenance, and human-review gates.
