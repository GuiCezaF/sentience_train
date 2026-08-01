# Reserve final acceptance for the complete export bundle

`export --format onnx` and `export --format tflite` are development operations
that validate only against `PublicTest` and do not mark a run as released.
`export --format all` is the default and the only release operation: it creates
both artifacts, evaluates the frozen bundle once on `PrivateTest`, applies all
quality and deployment gates, and may mark the run `accepted`. This prevents an
isolated API export from consuming final test access or yielding a partially
validated release.
