# Reserve PrivateTest for the frozen release candidate

Model candidates are trained with the `Training` partition and selected by
applying proxy thresholds to `PublicTest`. Architecture, hyperparameters, and
checkpoint are then frozen before one release candidate export bundle is
evaluated on `PrivateTest`. A failing bundle fails that release and must not be
replaced by choosing another candidate from the observed test results. This
keeps final acceptance distinct from model selection and makes test access
auditable.
