# EviAudio nested answer-development protocol

**Frozen before any Qwen2-Audio or Phi-4 answer was generated on the Clotho-derived compositions.** This stage uses only the already-declared 281-source exploratory resource. No AUDITA question, answer, waveform, output, or category result may be accessed.

## Purpose

The passed exact-onset experiment establishes only temporal event retrieval. Before a one-shot AUDITA answer evaluation can be authorized, the answer reservoir, prompt, retrieval family, and two audio-language backbones must work on source-held-out exploratory audio without label leakage. This protocol creates that bridge; it cannot establish the confirmatory answer claim.

## Source-level nesting

The 49 target sources in `journal_suite/data/manifests/val.jsonl` are sorted by SHA-256 of `eviaudio-answer-development-v1|target_source_id`. The first 24 are calibration and the remaining 25 are held-out evaluation. Every composition is assigned by its target source, so repeated questions and compositions from one target never cross the boundary. Distractor reuse is recorded; source-macro statistics use the target source as the unit.

Retrieval selection uses all calibration compositions and no generated answers. Fixed CLAP scores are the baseline. The five frozen `temp_010_ev010` QCR checkpoints and the five frozen exact-event checkpoints are each averaged without seed selection. Candidate rankers are ordered by target-source-macro source accuracy, then evidence AP, MRR, and lexical name. A learned ranker is admissible only if both source accuracy and evidence AP are no worse than CLAP; otherwise CLAP remains the deployed retriever and the learned-retrieval development gate fails.

After the retriever is fixed, prompt and reservoir size are selected on at most four deterministically hash-ranked compositions per calibration source. The grid contains exactly two frozen short-answer prompts and K in {2, 4, 6}. Chunks are four seconds, selected by score, then presented in original chronological order. The same prompt/K pair is shared by both backbones. It ranks by the mean of the two backbone-specific target-source-macro exact scores, then token F1, smaller K, and lexical prompt ID. Generation is greedy with at most 12 new tokens.

## Held-out development evaluation

The fixed retriever, prompt, and K are evaluated once on every composition whose target is among the 25 held-out sources. Systems are learned retrieval, CLAP, deterministic random, uniform, equal-duration prefix, oracle evidence, selected evidence replaced by silence, and text-only. Exact match and token F1 are primary development outcomes; source-clustered uncertainty, difficulty, stream length, target position, latency, peak memory, and generated-output hashes are mandatory.

The answer stage may be frozen for AUDITA only if learned retrieval improves source-macro exact match by at least two points over the strongest eligible non-oracle retrieval baseline and five points over random, neither backbone loses more than two points to the strongest baseline, both backbones produce nonempty audio-conditioned outputs, and every provenance, split, checksum, and alignment audit passes. This is an authorization threshold, not the final Q1 gate; AUDITA still requires a five-point gain with clustered significance and all intervention/category gates in `PREREGISTRATION.md`.

## Reproducibility and limits

Qwen2-Audio and Phi-4 Multimodal use the pinned revisions in `configs/answer_development.json`, BF16, SDPA, and a dedicated Transformers 4.48.2 runtime. Original exploratory WAV container bytes are materialized from pinned parquet only after matching all 281 stored hashes. No prompt or candidate may be added after the first generated answer. A failed held-out gate seals AUDITA and is reported as a negative result.
