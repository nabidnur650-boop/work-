# EviAudio held-out answer-evaluation protocol

**Frozen before the 25-source Clotho held-out answer set is opened.** The
retriever was selected without generated answers. The prompt and K will be
copied verbatim from the completed two-backbone calibration report; this
protocol supplies no alternate prompt, K, model, or gate.

## Evaluation unit and systems

All 411 compositions belonging to the 25 source-disjoint evaluation targets
are evaluated exactly once by each of two pinned audio-language backbones.
The eight systems and their exact chunk-selection rules are recorded in
`configs/answer_heldout.json`. Selected chunks are always concatenated in
chronological stream order. Every audio system is bounded by the selected K
four-second chunks. Oracle retrieval may use evidence labels only to estimate
the answer-model ceiling and is ineligible as a comparator.

The strongest eligible non-oracle baseline is selected conservatively on the
held-out results from CLAP, deterministic random, uniform, and equal-duration
prefix retrieval. It ranks by the two-backbone mean target-source-macro exact
score, token F1, then lexical system name. Silence and text-only are causal
controls, not eligible retrieval baselines.

## Frozen development gates

The learned system must improve two-backbone mean source-macro exact match by
at least 0.02 over the strongest eligible baseline and 0.05 over random.
Neither backbone may trail the strongest baseline by more than 0.02. Every
response must be nonempty. For each backbone, at least 20% of normalized
learned responses must differ from the matched-silence responses and at least
20% must differ from text-only; learned exact match must be no worse than
silence. All provenance, checksum, split, alignment, duration, and job-count
checks must pass.

Target source is the resampling unit for 10,000 fixed-seed paired bootstrap
replicates. These intervals are mandatory descriptive uncertainty checks but
are not added post hoc to the previously frozen development threshold.
Difficulty, number of sources, target-position bin, and chunk-count quartile
are reported without suppressing unfavorable strata.

## Claim boundary

Passing authorizes one separately locked AUDITA run; it is not confirmatory
evidence itself. Failure permanently seals AUDITA for this pipeline. No
AUDITA question, answer, waveform, model output, or category result may be
accessed during this stage.
