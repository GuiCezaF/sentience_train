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

    lines.append("## Metrics (from this test session)")
    lines.append("")

    try:
        from evaluation import (
            build_metrics_summary,
            format_metrics_summary_markdown,
            load_evaluation_metrics_json,
            validate_evaluation_metrics,
        )
    except ImportError as exc:
        lines.append(
            "_Could not import `evaluation` (missing dependencies?): "
            f"{exc!r}_"
        )
    else:
        generated = ROOT / "reports" / "evaluation_metrics.generated.json"

        if not generated.is_file():
            lines.append(
                "_No session metrics JSON yet: `reports/evaluation_metrics.generated.json` "
                "is produced when you run `tests/test_evaluation_metrics.py`._"
            )
        else:
            doc = load_evaluation_metrics_json(generated)
            errs = validate_evaluation_metrics(doc)
            try:
                rel = generated.relative_to(ROOT)
            except ValueError:
                rel = generated
            lines.append(f"**Source:** `{rel}`")
            meta = ROOT / "reports" / "evaluation_run_meta.txt"
            if meta.is_file():
                lines.append("")
                lines.append(f"**Run metadata:** `{meta.relative_to(ROOT)}`")
                lines.append("")
                lines.append("```")
                lines.append(meta.read_text(encoding="utf-8").rstrip())
                lines.append("```")
            lines.append("")
            if errs:
                lines.append("**Structural validation:** failed.")
                for e in errs:
                    lines.append(f"- {e}")
            else:
                lines.append("**Structural validation:** OK.")
                lines.append("")
                lines.append(
                    format_metrics_summary_markdown(build_metrics_summary(doc))
                )

    out = REPORTS_DIR / "test_report.md"
    out.write_text("\n".join(lines), encoding="utf-8")
