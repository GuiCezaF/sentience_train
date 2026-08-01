from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from sentience_train.export import export_frozen_run


@pytest.mark.acceptance
def test_real_frozen_run_release_has_honest_final_status() -> None:
    run_id = os.getenv("SENTIENCE_RELEASE_RUN_ID")
    if not run_id:
        pytest.fail("SENTIENCE_RELEASE_RUN_ID is required for real acceptance")
    runs_root = Path(os.getenv("SENTIENCE_RELEASE_RUNS_ROOT", "runs"))
    exports_root = Path(os.getenv("SENTIENCE_RELEASE_EXPORTS_ROOT", "exports"))

    result = export_frozen_run(run_id, runs_root=runs_root, exports_root=exports_root)

    assert result.status in {"accepted", "failed"}
    manifest = json.loads((result.directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == result.status
    assert manifest["provenance"]["dataset"]["canonical_sha256"]
    if result.status == "accepted":
        assert manifest["gates"]["passed"] is True
    else:
        assert manifest["gates"]["passed"] is False
