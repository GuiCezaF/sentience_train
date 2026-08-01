# Package the TFLite model for firmware handoff

The deployment handoff contains `emotion_model_int8.tflite`, aligned const C++
array files `emotion_model_data.h` and `.cc`, canonical `labels.txt`, an
`input_contract.json` describing shape, type, layout, scale, and zero point, and
an `operators.json` enumerating the exact TFLite Micro resolver allowlist. The
common manifest carries artifact hashes. The service handoff contains
`emotion_model_fp32.onnx` and its input/output contract. ESP-IDF firmware,
camera capture, face detection, and alignment remain outside this repository.
