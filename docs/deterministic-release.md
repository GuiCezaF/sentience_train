# Deterministic release

The public release workflow is:

```text
sentience-train export --run-id <run-id>
```

`all` is the default and publishes to `exports/<run-id>/all/`, with ONNX FP32
and TFLite INT8 handoffs, `PublicTest` validation, host gates, and only then the
atomic reservation of `PrivateTest`. The `onnx` and `tflite` formats publish
individual handoffs and never create the reserved-access ledger.

ONNX records NCHW `[N,1,48,48]`, a dynamic batch, and
`pixel / 127.5 - 1` normalization. TFLite records NHWC `[1,48,48,1]`, a fixed
batch, scale, zero point, labels, operators, and C++ files derived from the same
bytes.

The complete transaction reserves `PrivateTest` exactly once under
`exports/<run-id>/.private-test-access/<candidate-identity>/`. A crash, a
concurrent reservation, or a ledger without a final bundle blocks another
attempt. Finalized bundles are immutable and may be reused only after the
manifest digest and every payload have been validated.

Host gates enforce 1 MiB for TFLite, 1.5 MiB for the arena estimate with
margin, and 15 million MACs. The ESP32-S3 arena measurement and a p95 latency
of 200 ms remain `pending_external`. The final status is always `accepted` or
`failed`; `PublicTest` and `PrivateTest` results remain separate.
