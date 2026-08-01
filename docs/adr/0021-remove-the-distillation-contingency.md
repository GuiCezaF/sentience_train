# Remove the distillation contingency

**Status:** Accepted

No evidence shows that the MiniVGG/distillation fallback was used. New
training therefore stops with immutable `SelectionFailed` diagnosis when all
DS-CNN widths and seeds fail. It does not build a teacher, retry distilled
students, weaken gates, or read `PrivateTest`. This decision supersedes ADR
0019. The export resolver remains backward-compatible with valid
`frozen-run/v1` artifacts whose historical manifests contain teacher metadata.
