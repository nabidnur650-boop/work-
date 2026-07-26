# Public-path sanitization

Some frozen runtime records originally contained machine-local absolute
paths. Public copies normalize project-root prefixes to repository-relative
paths and replace any remaining home prefix with `<LOCAL_HOME>`. Numerical
values, predictions, decisions, and scientific hashes are unchanged.

`SCIENTIFIC_ARTIFACT_INVENTORY.json` records the original and public SHA-256
values for every transformed file. The original source-tree records remain
untouched, so their hashes continue to match the frozen authorization and
lock documents.
