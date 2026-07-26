# Held-out answer uncertainty supplement

**Frozen after held-out execution was authorized but before any held-out
response or aggregate result was inspected.** This supplement changes no
system, prompt, K, retriever, metric definition, comparator selection, or
promotion gate.

The primary analyzer already computes paired target-source bootstrap
intervals for exact-match differences. To give the co-primary descriptive
token-F1 outcome equal uncertainty coverage, this supplement additionally
reports 10,000-replicate target-source percentile intervals for:

- absolute source-macro exact match and token F1 for every model/system;
- two-backbone mean source-macro exact match and token F1 for every system;
- paired learned-minus-comparator token-F1 differences for all seven
  comparators, per backbone and combined.

The source is the only resampling unit. The seed is 20260722. These intervals
are descriptive and cannot rescue or fail the frozen development gates.
AUDITA remains sealed.
