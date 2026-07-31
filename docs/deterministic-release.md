# Deterministic release guide

The official release command is:

```text
sentience-train export --run-id <run-id>
```

`all` is the default and publishes `exports/<run-id>/all/`. It validates the
immutable frozen run, creates the ONNX and copied INT8 TFLite handoffs, and
records PublicTest evidence before the final release gate.

The `onnx` and `tflite` formats are development exports. They are limited to
the indexed `PublicTest` partition and never create a PrivateTest ledger.

The complete flow reserves PrivateTest exactly once at:

```text
exports/<run-id>/.private-test-access/<candidate-identity>/
```

The directory is created atomically before any reserved row is read. A crash,
concurrent reservation, or ledger without a final bundle blocks retry. A
completed bundle is immutable and may be reused only after its root manifest
self-digest and all payload hashes validate.

Candidate identity is derived from the frozen manifest, checkpoint, TFLite,
dataset, and frozen configuration. Exporter source changes and lock changes
are provenance changes; they cannot authorize a second PrivateTest evaluation.

Every complete bundle ends in `accepted` or `failed`. The final evidence keeps
PublicTest and PrivateTest separate, stores metrics and raw confusion matrices,
and marks ESP32-S3 latency p95 and measured arena as `pending_external` until
target-device evidence is supplied.
