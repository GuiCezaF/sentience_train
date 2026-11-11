from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.mark.acceptance
def test_real_acceptance_requires_explicit_run() -> None:
    run_id = os.environ.get("SENTIENCE_RELEASE_RUN_ID")
    if not run_id:
        pytest.fail("SENTIENCE_RELEASE_RUN_ID is required for real acceptance")
    assert Path(run_id).name == run_id
    from sentience_train.export import export_frozen_run

    result = export_frozen_run(run_id, config=os.environ.get("SENTIENCE_RELEASE_CONFIG", "configs/default.yaml"))
    assert result.status == "accepted"
    manifest = __import__("json").loads((result.directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["private_test_accessed"] is True
    benchmark = __import__("json").loads((result.directory / "benchmark-summary.json").read_text(encoding="utf-8"))
    assert benchmark["cpu_normalized"] == "AMD Ryzen 7 5700"
    assert (
        manifest["onnx_sha256"]
        == __import__("hashlib").sha256((result.directory / "model.onnx").read_bytes()).hexdigest()
    )
