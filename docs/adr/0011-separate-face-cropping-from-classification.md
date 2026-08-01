# Accept face crops through artifact-specific input contracts

Face detection, alignment, capture, and full-frame camera processing remain
outside this classifier repository. The TFLite deployment model accepts one
INT8 NHWC tensor shaped `[1, 48, 48, 1]`, with its quantization scale and zero
point published in export metadata. The ONNX service model accepts normalized
FP32 NCHW tensors shaped `[N, 1, 48, 48]` and permits dynamic API batch size.
Both wrappers implement preprocessing equivalent to grayscale normalization in
`[-1, 1]`, and export validation compares their predictions for the same face
crops against the PyTorch checkpoint.
