# Target an N8R8-class ESP32-S3-CAM with a bounded PSRAM budget

The deployment target is an ESP32-S3-CAM N8R8 or equivalent with 8 MB flash
and 8 MB PSRAM. The fully INT8 `.tflite` artifact must occupy at most 1 MiB in
flash, and its tensor arena must occupy at most 1.5 MiB in PSRAM; export reports
must record both values. We allow external PSRAM because restricting inference
to the ESP32-S3's shared internal SRAM would impose a substantially harsher
capacity trade-off against the 70% accuracy requirement. Camera and complete
firmware memory usage remain integration checks for the separate embedded
project.
