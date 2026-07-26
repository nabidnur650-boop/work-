# Cross-fitted natural-audio routing analysis

This track does not modify the frozen Perception Test result.  That result
remains a failed zero-shot transfer test for the fixed QCR multiscale method.

The new analysis asks a different question: after that failure, can a small
amount of in-domain supervision identify which already-computed score source
and temporal scale should be used for each class?  Five deterministic
video-disjoint folds are used.  For each held-out fold and class, the router
selects one candidate using only the other four folds.  Held-out predictions
are concatenated once and evaluated with the pinned official metric.

This is a post-outcome, cross-fitted exploratory analysis—not a fresh
preregistration and not a zero-shot result.  Its purpose is to localize the
failure mechanism and test a practical low-capacity repair without training a
new audio encoder.
