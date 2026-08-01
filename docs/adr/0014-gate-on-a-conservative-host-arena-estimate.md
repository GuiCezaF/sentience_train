# Gate on a conservative host tensor-arena estimate

Export runs a C++ `RecordingMicroInterpreter` probe built against a pinned
TFLite Micro version and an operator resolver containing only the deployment
model's operations. The report records head, temporary, tail, and total arena
allocations, adds 20% safety headroom, and applies the 1.5 MiB gate to that
conservative value. It labels the result `estimated_tflm_arena_bytes`; only the
separate embedded integration may populate `measured_esp32s3_arena_bytes`.
This provides an automated pre-deployment memory check without presenting host
behavior as target-device evidence.
