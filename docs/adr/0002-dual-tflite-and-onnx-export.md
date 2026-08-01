# Export TFLite for the microcontroller and ONNX for the API

Each accepted training checkpoint produces an export bundle containing a TFLite
deployment model for TFLite Micro with ESP-NN and an ONNX service model for the
separate API. Both artifacts must be checked against the same PyTorch reference
outputs. The service model remains FP32 and accepts dynamic API batch size;
ONNX quantization is outside this refactoring. TFLite is the artifact governed
by the embedded memory budget, while ONNX remains a supported API artifact
rather than an intermediate required to produce TFLite; this avoids coupling
microcontroller delivery to a longer ONNX-to-TensorFlow conversion chain
without removing the existing API use case.
