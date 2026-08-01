# Apply the predictive quality gate to the quantized deployment model

An export bundle is accepted only when its fully INT8 TFLite deployment model
reaches at least 70% accuracy and 0.65 macro-F1 on the filtered FER2013
`PrivateTest` partition, every target emotion reaches at least 0.55 recall, and
quantization loses no more than two percentage points of accuracy relative to
the PyTorch checkpoint. The ONNX service model must remain equivalent to the
same PyTorch checkpoint. These additional thresholds prevent aggregate
accuracy from hiding a weak emotion class or an unusable quantized artifact.
