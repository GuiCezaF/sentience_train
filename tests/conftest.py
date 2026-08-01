from __future__ import annotations

import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = ROOT / "reports"
_TEST_ROWS: list[tuple[str, str, float]] = []


def pytest_sessionstart(session: pytest.Session) -> None:
    _TEST_ROWS.clear()


def pytest_configure(config: pytest.Config) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.when == "call":
        dur = float(report.duration or 0.0)
        _TEST_ROWS.append((report.nodeid, report.outcome or "unknown", dur))


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    lines = [
        "# Test report",
        "",
        f"**UTC:** {datetime.datetime.now(datetime.timezone.utc).isoformat()}",
        f"**Pytest exit code:** {exitstatus}",
        "",
        "## Test cases",
        "",
        "| Test | Outcome | Duration (s) |",
        "|------|---------|----------------|",
    ]
    for nodeid, outcome, dur in _TEST_ROWS:
        safe = nodeid.replace("|", "\\|")
        lines.append(f"| `{safe}` | {outcome} | {dur:.4f} |")
    lines.extend(
        [
            "",
            "## JUnit XML",
            "",
            "File written by pytest: `reports/junit.xml`.",
            "",
        ]
    )

    out = REPORTS_DIR / "test_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
