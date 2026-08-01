# Bound inference compute and defer hardware latency measurement

Every model candidate is limited to 15 million multiply-accumulate operations
for one 48x48 grayscale input. The embedded integration target is p95 inference
latency at or below 200 ms on an ESP32-S3 running at 240 MHz with ESP-NN. This
repository must calculate and report operation count, while the separate
embedded project must measure device latency; a host estimate must never be
reported as an on-device result.
