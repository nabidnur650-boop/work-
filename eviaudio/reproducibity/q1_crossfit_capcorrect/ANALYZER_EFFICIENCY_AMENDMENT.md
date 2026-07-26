# Analyzer efficiency amendment before router scoring

After the cap-corrected protocol was frozen, code review found that one
postcondition counted output rows separately for every video. That check was
quadratic in videos times predictions and would be impractically slow.

Before candidate-pool completion or any router selection/result, the
postcondition was replaced by a single `Counter` pass over the same output
rows. The routing, sorting, caps, candidate selection, metrics, bootstrap,
diagnostic criteria, and all numerical parameters are unchanged. The amended
analyzer verifies every originally locked file except the superseded analyzer
hash and additionally verifies the amendment lock.
