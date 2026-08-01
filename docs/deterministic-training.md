# Deterministic FER2013 training

The public workflow is `sentience-train train --config PATH`. It reads only
`Training` for optimization and `PublicTest` for validation, early stopping,
calibration, and selection. `PrivateTest` remains only in the CSV provenance
and is not materialized by the engine.

Selection evaluates DS-CNN `w025`, `w035`, and `w050` with seeds `42,43,44`,
stopping at the first approved width. PTQ precedes QAT; QAT responds only to
conversion-failure or accuracy-loss triggers. New runs have no teacher or
distillation. If every candidate fails, the immutable run ends with
`selection_failed`, a complete diagnostic, and no `PrivateTest` access.

The input is the canonical FER2013 CSV, validated by schema and SHA-256; the
output classes are `angry,happy,neutral,sad`, with the raw mapping `0,3,6,4`.
Normalization is `pixel / 127.5 - 1`, and class weights use inverse square-root
frequency normalized by the mean.

Each run contains the resolved configuration, a frozen `PublicTest` index,
checkpoints, history, metrics, candidate comparisons, hashes, and a
`frozen-run/v1` manifest. Valid earlier runs remain consumable by the export,
even if they record historical teacher metadata.
