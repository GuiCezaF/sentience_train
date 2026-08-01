# Preflight direct TFLite export before training

Before any full candidate training, the pipeline instantiates each configured
DS-CNN width and exercises direct PyTorch-to-LiteRT conversion, full INT8 I/O,
operator inspection, model and arena budgets, and operation counting. A failed
direct conversion stops with a diagnostic before expensive training begins.
The pipeline must not silently switch to an ONNX-to-TensorFlow conversion path;
introducing such a fallback requires an explicit new decision because it adds a
second converter chain and reproducibility surface.
