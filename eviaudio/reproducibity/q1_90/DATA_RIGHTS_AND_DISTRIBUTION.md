# Data rights and release boundary

The Q1-90 replication uses only public research datasets with explicit source
records. It does not use AUDITA outcomes, questions, answers, or audio. The
revised claim is limited to source-disjoint acoustic evidence localization.

## Sources

- **ESC-50** is the development foreground collection. Its official repository
  states that ESC-50 is available under a Creative Commons Attribution
  Non-Commercial license and provides clip-level attributions:
  <https://github.com/karolpiczak/ESC-50>. Original Freesound identifiers from
  its metadata are used to exclude overlapping UrbanSound8K sources.
- **UrbanSound8K 1.0.0** is the external foreground collection. The official
  Zenodo record, DOI `10.5281/zenodo.1203745`, declares CC BY-NC 4.0 and MD5
  `9aa69802bbf37fb986f71ec1483a196e` for the 6.0-GB archive:
  <https://zenodo.org/records/1203745>.
- **LibriSpeech SLR12** supplies background speech. OpenSLR declares CC BY 4.0;
  the external study uses only `test-other`, whose official checksum is
  `fb5a50374b501bb3bac4815ee91d3135`:
  <https://www.openslr.org/12>.

## Distribution rule

Raw archives, extracted waveforms, and third-party audio are excluded from the
submission/reproduction archive. The release contains download instructions,
official URLs and checksums, deterministic source-level recipes, source hashes,
code, aggregate results, and integrity manifests. Any local archive or derived
embedding cache is a non-redistributed runtime artifact. Users must obtain the
datasets from their official hosts and comply with their licenses, attribution
requirements, and non-commercial restrictions.
