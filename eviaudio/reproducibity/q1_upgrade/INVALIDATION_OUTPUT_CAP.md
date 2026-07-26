# Invalidation: the first cross-fit analyzer did not reapply the video cap

The first locked analyzer was stopped before it wrote router predictions or a
router report. Its `assemble_predictions` function joined class-specific rows
from different candidate files but did not reapply the natural-panel limit of
200 predictions per video.

That omission matters because each source candidate had been capped to 200
predictions only *before* class-wise routing. Combining classes from different
candidates can exceed that limit and can also retain a different set of rows
than a composite router followed by one shared cap. Any router result from
that implementation would therefore be incomparable with the frozen CLAP
multiscale baseline.

The run was interrupted while candidate-selection scores were still being
computed. It created no prediction file and no report. The following
full-panel descriptive candidate mAP values had appeared in stdout before the
defect was found:

- `clap_0p5s`: 0.061454651593559985
- `clap_1s`: 0.07269542139116228
- `clap_2s`: 0.06972355442556935
- `clap_4s`: 0.05228012871515174
- `clap_multiscale`: 0.06757634351816902
- `qcr_0p5s`: 0.051064981030764046

These exposed values are disclosed and are not treated as fresh evidence.
The corrected analysis is isolated in `q1_crossfit_capcorrect`. It regenerates
per-class pools with the unchanged 50-segment class cap, selects candidates
out of fold, merges the selected classes, and only then applies the unchanged
200-segment video cap. The follow-up remains explicitly post-outcome and
exploratory.
