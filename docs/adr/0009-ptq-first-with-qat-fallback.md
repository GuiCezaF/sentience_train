# Use post-training quantization before quantization-aware training

Each FP32 model candidate is first fully quantized to INT8 using a deterministic
and target-emotion-stratified calibration sample of 512 unaugmented examples
from the `Training` partition. We introduce quantization-aware training only
when the FP32 candidate passes the validation gate but its INT8 deployment model
loses more than two percentage points of accuracy or fails another validation
threshold. This keeps the normal training path simple while retaining a bounded
recovery path for quantization loss; validation and test examples are never used
for calibration.
