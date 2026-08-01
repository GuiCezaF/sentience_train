# Use train and export as the only public commands

**Status:** Superseded by [ADR 0020](0020-reserve-final-acceptance-for-the-complete-export-bundle.md).

The repository now exposes only `train` and `export`; this decision remains the
CLI boundary. ADR 0020 is the current authority for which export format may
perform final acceptance and consume `PrivateTest`.
