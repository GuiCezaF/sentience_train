# Explore a bounded depthwise-separable model family

Model candidates use a MobileNetV1-inspired DS-CNN composed of an initial
convolution, depthwise and pointwise blocks, export-foldable batch
normalization, ReLU6, strided downsampling, fixed global average pooling, and a
four-output linear classifier. The search is limited to width multipliers 0.25,
0.35, and 0.50; candidates are tried from smallest to largest, and the process
stops at the first candidate that satisfies the validation gate. Architectures
with large dense layers or residual branches are excluded to keep parameters,
activation memory, operation count, and TFLite Micro operator requirements
predictable. A candidate that statically exceeds a deployment constraint is not
trained.
