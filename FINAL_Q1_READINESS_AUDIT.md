# Final Q1-readiness audit

## Outcome

Both research packages are technically ready for GitHub upload and
accountable-author review. ShiftTitan scores 8.71/10 and EviAudio 8.95/10 on
the declared package-readiness rubric. All 169 source-tree tests pass, both
manuscript/supplement pairs render inside current internal venue guards, and
both ZIP archives reproduce bit-for-bit across consecutive builds.

The objective technical GitHub gate is 10/10: manifest integrity, repository
and per-file size, archive CRC/determinism, tests, numerical recomputation,
portable paths, credential-pattern hygiene, staging completeness, and branch
state all pass. The repository interface also includes immutable action SHAs,
least-privilege CI, pinned verifier dependencies, validated navigation,
issue/PR templates, and explicit contribution/security policies. This does
not replace human authorization: author identity, copyright ownership,
repository ownership, and final disclosures remain pending.

## Publication artifacts

- ShiftTitan: 14-page main manuscript, 5-page supplement, four main figures,
  three main tables, 177-word abstract, and 68 scientific reproduction files.
- EviAudio: 6-page main manuscript, 3-page supplement, four main figures, two
  main tables, 193-word abstract, and 232 scientific reproduction files.
- Every publication figure is bound to machine-readable source data and
  SHA-256 values.
- Public metadata copies contain no machine-local user-home paths; every
  normalized file records both its frozen source hash and public-copy hash.
- Candidate pools and third-party datasets/weights are excluded; hashes,
  receipts, recipes, final predictions, and reconstruction code remain.

## Independent numerical recomputation

- Both ShiftTitan task-by-backbone matrices, effects, counts, and task
  bootstraps recompute from retained summaries and 99,186 raw origin rows.
- Controlled/external audio metrics and hierarchical bootstraps recompute
  from 616 raw examples.
- The natural per-video endpoint recomputes across all 5,359 videos.
- All 5,449,000 retained natural/router predictions pass hash, schema,
  finiteness, coverage, and cap checks.
- All 6,576 held-out answer rows reproduce exact/F1, source-macro effects,
  gates, and paired bootstraps.
- Natural/router official global mAP arithmetic is verified from retained
  class-by-threshold arrays. A fresh ground-truth recomputation correctly
  remains outside the compact release because provider annotations are not
  redistributed.

## Scientific integrity

- ShiftTitan preserves both failed frozen promotion decisions and labels the
  pooled/family analyses post hoc.
- EviAudio preserves the failed fixed natural-transfer and downstream
  decisions and labels the cap-corrected router post outcome.
- Both invalidated predecessor analyses remain documented.
- No threshold, result, identity, legal declaration, or acceptance guarantee
  was fabricated.

## Remaining human-only work

Complete author order, affiliations, ORCIDs, CRediT roles, corresponding
contact, funding, conflicts, acknowledgments, copyright/data/model rights,
journal-required disclosures, final proofreading, repository ownership, and
the authenticated commit/push.
