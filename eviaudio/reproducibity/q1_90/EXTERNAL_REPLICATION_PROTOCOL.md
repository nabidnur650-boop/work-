# EviAudio Q1-90 zero-shot external replication

## Claim and boundary

This follow-up tests only the supported exact-onset evidence-localization
claim. It does not reinterpret the failed Clotho answer bridge and does not
open AUDITA. The five ESC-50-trained residual rankers, their ensemble rule,
the pinned CLAP model, 4 s chunks, 2 s hop, and all scoring code remain frozen.

The external foreground corpus is UrbanSound8K. To prevent cross-dataset
contamination through their shared Freesound origin, every UrbanSound8K
`fsID` appearing in ESC-50 metadata is excluded before sampling. Fresh
LibriSpeech `test-other` speakers provide background audio; development used
`dev-clean` and the earlier confirmation used `test-clean`.

## Deterministic construction

- UrbanSound8K folds 8--10 are the target pool. The original folds 9--10
  pool failed a model-blind metadata feasibility check because jackhammer had
  only eight ESC-disjoint original source IDs. Before any recipe or model
  output existed, the smallest adjacent nested expansion (adding fold 8) was
  frozen; it raises the minimum to 12. The complete count-only amendment is
  recorded in `MODEL_BLIND_SOURCE_COUNT_AMENDMENT.json`.
- At most 25 target clips per class are selected after source-ID exclusion,
  with one target per original `fsID` and at least 12 required per class.
- Target clips are never reused as distractors. Three other-class distractors
  are chosen deterministically from non-target sources in the same fold.
- Duration (60/180/300 s), target SNR (-5/0/5 dB), position
  (early/middle/late), and one of three query templates are fixed by SHA-256.
- Mixtures are generated in memory. No long waveform or third-party source
  audio is redistributed in a submission archive.

Metadata and waveform integrity may be inspected while constructing recipes.
No CLAP embedding, learned score, or performance outcome may be computed until
the recipe audit, code, checkpoints, gates, and authorization are hashed.

## Outcomes and gates

The primary outcomes are evidence average precision and exact-onset Hit@1.
Uncertainty uses 10,000 fixed-seed paired bootstrap replicates over original
UrbanSound `fsID`; repeated conditions remain paired. All gates must pass:

1. ensemble AP gain over the frozen CLAP prior is at least 0.05;
2. ensemble Hit@1 gain is at least 0.05;
3. both paired 95% bootstrap lower bounds are above zero;
4. at least four of five frozen seeds improve both outcomes;
5. no class loses more than 0.05 Hit@1;
6. the external performance index, `50 * (AP + Hit@1)`, is at least 85;
7. recipe, source-isolation, archive, alignment, residual-bound, and checksum
   audits all pass.

The complete report is retained if any gate fails. No parameter, recipe,
prompt, subset, or gate may be changed after scoring.

## Provenance

- UrbanSound8K v1.0: DOI `10.5281/zenodo.1203745`, archive MD5
  `9aa69802bbf37fb986f71ec1483a196e`, non-commercial attribution license.
- LibriSpeech SLR12 `test-other`: CC BY 4.0, archive MD5
  `fb5a50374b501bb3bac4815ee91d3135`.
- CLAP: `laion/clap-htsat-unfused`, revision
  `8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a`.

Audio remains outside the distributable package. The release includes only
acquisition instructions, source identifiers, checksums, recipes, derived
metrics where permitted, and executable verification.
