# Frozen held-out answer figure protocol

**Frozen on 2026-07-22 after held-out execution was authorized and before any
held-out response, metric, or aggregate result was inspected.** Figure inclusion
does not depend on whether the promotion gate passes. The suite reports the one
authorized run and cannot change a model, prompt, K, retriever, metric,
comparator, bootstrap, or promotion threshold.

The suite will contain the following 40 source-linked figures:

1. evaluation path and separation of calibration, held-out development, and sealed AUDITA;
2. combined exact match for all eight systems;
3. combined token F1 for all eight systems;
4–5. per-backbone exact match and token F1;
6. paired source-bootstrap exact differences to all comparators;
7. paired source-bootstrap token-F1 differences to all comparators;
8. normalized-response differences from silence and text-only controls;
9. all frozen promotion-gate outcomes;
10. calibration versus held-out selected-system metrics;
11–14. source-level selected-system exact, token F1, exact difference to the
strongest eligible baseline, and exact difference to deterministic random;
15–18. selected-system exact and token-F1 breakdowns by difficulty and number
of composed sources;
19–22. selected-system exact and token-F1 breakdowns by target-position bin and
chunk-count quartile;
23–24. absolute source-bootstrap intervals for exact match and token F1;
25–26. per-example exact transitions versus the strongest eligible baseline and
deterministic random;
27–28. selected-system audio duration and selected-positive-chunk distributions;
29–30. generation latency and input-token distributions by backbone/system;
31. response-token-count distribution by backbone/system;
32. examples per held-out target source;
33. oracle headroom for each backbone and their mean;
34. selected-system differences from silence and text-only exact match;
35. evidence yield across all audio systems;
36. calibration prompt/K candidate surface;
37. backbone runtime and memory inventory from immutable receipts;
38. integrity inventory;
39. source-level selected-versus-oracle exact difference;
40. gate-relevant effect-size summary with frozen thresholds.

Every figure must be emitted as vector PDF and 300-dpi PNG with a dedicated CSV
source table, SHA-256 hashes, a unique identifier, and an indexed caption.
AUDITA rows are forbidden. The claim scope is the held-out Clotho answer gate
only; a pass merely makes a separately locked AUDITA pipeline eligible.
