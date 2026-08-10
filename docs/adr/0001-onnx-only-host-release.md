# ONNX-only host release

Status: Accepted.

The project publishes one FP32 ONNX artifact and excludes embedded-runtime
formats from the public pipeline. This gives every consumer the same bytes and
places normalization and layout conversion inside a versioned graph contract.
The benchmark and release publisher share the staged path, so a validated file
cannot differ from the published file.
