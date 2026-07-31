# Facial Emotion Recognition

This context defines the language used to train and deliver a facial-emotion
classifier for constrained embedded inference.

## Language

**Target emotion**:
One of the four facial-expression labels recognized by this project: Angry,
Happy, Neutral, or Sad.
_Avoid_: Supported class, selected class

**Face crop**:
A single already-located facial image represented as 48x48 grayscale pixels and
ready for emotion classification.
_Avoid_: Camera frame, face detection result when it has not been normalized

**Training partition**:
The target-emotion examples marked `Training` in the canonical FER2013 corpus.
_Avoid_: Training folder

**Validation partition**:
The target-emotion examples marked `PublicTest` in the canonical FER2013
corpus, used for model selection and early stopping.
_Avoid_: Public test, test set

**Calibration sample**:
A deterministic, emotion-stratified subset of the training partition used only
to estimate integer quantization ranges.
_Avoid_: Validation sample, calibration set drawn from test data

**Test partition**:
The target-emotion examples marked `PrivateTest` in the canonical FER2013
corpus, reserved for final measurement after model selection.
_Avoid_: Validation set, evaluation set

**Deployment model**:
The fully quantized classifier artifact accepted for execution on the target
microcontroller, mapping a 48x48 grayscale face crop to one target emotion.
_Avoid_: Final model, edge model

**Service model**:
The classifier artifact accepted for inference by the separate application API
and derived from the same checkpoint as the deployment model.
_Avoid_: API deployment model, final model

**Export bundle**:
The deployment model and service model produced and validated together from one
accepted training checkpoint.
_Avoid_: Model export when referring to only one artifact

**Accepted checkpoint**:
A trained model state whose export bundle satisfies the agreed predictive
quality and deployment constraints on the reserved test partition.
_Avoid_: Best checkpoint when it has only the highest validation score

**Model candidate**:
A model architecture and training configuration evaluated as one option for an
accepted checkpoint.
_Avoid_: Experiment when referring to the resulting model option

**Release candidate**:
The model candidate frozen after validation-based selection and eligible for
final measurement on the reserved test partition.
_Avoid_: Accepted checkpoint before the quality gate has passed

**Validation gate**:
The quality-gate thresholds applied to the validation partition as model
selection proxies; passing them does not establish final acceptance.
_Avoid_: Quality gate, test gate

**Quality gate**:
The complete set of predictive thresholds that an export bundle must pass,
including aggregate, per-emotion, and post-quantization requirements.
_Avoid_: Accuracy target

**Memory budget**:
The separate storage and runtime-memory limits that the deployment model must
satisfy on the target microcontroller.
_Avoid_: Model size, RAM budget when referring to both limits

**Estimated tensor arena**:
The deployment model's runtime working-memory requirement inferred outside the
target device and explicitly distinguished from an on-device measurement.
_Avoid_: Measured RAM, ESP32-S3 memory usage

**Measured tensor arena**:
The deployment model's runtime working-memory use observed on the target
ESP32-S3 integration.
_Avoid_: Host estimate, model file size

**Compute budget**:
The maximum operation count permitted for one deployment-model inference,
paired with a separately measured target-device latency objective.
_Avoid_: Performance when referring only to model operation count
