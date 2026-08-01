# Replace the legacy pipeline without compatibility shims

After the new two-command workflow passes its unit and integration tests, the
repository removes MiniVGGNet, BaselineCNN, the ImageFolder loader, global
import-time configuration, the old entrypoint, and the evaluate, crossval,
baseline, and compare pipelines. Useful metric calculations are migrated into
the new package before deletion. We deliberately avoid parallel legacy and new
implementations because their different data splits, models, and evaluation
semantics would invite undocumented and test-leaking behavior; the breaking
change is recorded in the changelog and README.
