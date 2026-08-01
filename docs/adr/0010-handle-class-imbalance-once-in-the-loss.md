# Handle class imbalance once in the loss

Training uses ordinary deterministic shuffling and a cross-entropy loss whose
class weights are computed only from `Training`, proportional to the inverse
square root of each target emotion's frequency and normalized to mean one.
Weighted sampling and oversampling are excluded, and reported metrics retain
the original validation and test distributions. This applies one moderate,
auditable correction instead of duplicating minority examples or accidentally
combining two balancing mechanisms as earlier code paths did.
