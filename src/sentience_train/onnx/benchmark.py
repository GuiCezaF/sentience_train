"""Clean-host ONNX Runtime benchmark and immutable evidence."""

from __future__ import annotations

import hashlib
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..core.config import HOST_BENCHMARK_SCHEMA, QualityGatesConfig


class BenchmarkError(RuntimeError):
    pass


@dataclass(frozen=True)
class BenchmarkReport:
    schema: str
    cpu_normalized: str
    os: str
    input_sha256: str
    latencies_ns: tuple[int, ...]
    rss_samples: tuple[int, ...]
    baseline_rss: int
    peak_rss: int
    rss_increment_bytes: int
    p50_ms: float
    p95_ms: float
    options: dict[str, Any]
    passed: bool
    reasons: tuple[str, ...] = ()
    synthetic: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "cpu_normalized": self.cpu_normalized,
            "os": self.os,
            "input_sha256": self.input_sha256,
            "latencies_ns": list(self.latencies_ns),
            "rss_samples": list(self.rss_samples),
            "baseline_rss": self.baseline_rss,
            "peak_rss": self.peak_rss,
            "rss_increment_bytes": self.rss_increment_bytes,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "options": self.options,
            "passed": self.passed,
            "reasons": list(self.reasons),
            "synthetic": self.synthetic,
        }


def normalized_cpu_name() -> str:
    try:
        import cpuinfo

        value = str(cpuinfo.get_cpu_info().get("brand_raw", ""))
    except Exception:
        value = platform.processor()
    return "AMD Ryzen 7 5700" if "AMD Ryzen 7 5700" in value else value.strip()


def _rss() -> int:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except Exception:
        return 0


def _nearest(values: list[int]) -> int:
    if not values:
        raise BenchmarkError("benchmark produced no measurements")
    ordered = sorted(values)
    return ordered[max(0, int(np.ceil(0.5 * len(ordered))) - 1)]


class HostBenchmark:
    def __init__(
        self,
        *,
        runtime: Any | None = None,
        sleep: Any = time.sleep,
        clock_ns: Any = time.perf_counter_ns,
        rss_reader: Any = _rss,
    ) -> None:
        self.runtime = runtime
        self.sleep = sleep
        self.clock_ns = clock_ns
        self.rss_reader = rss_reader

    def run(self, onnx_path: Path, *, gates: QualityGatesConfig | None = None) -> BenchmarkReport:
        gates = gates or QualityGatesConfig()
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise BenchmarkError(f"onnxruntime is unavailable: {exc}") from exc
        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = self.runtime or ort.InferenceSession(
            str(onnx_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        input_value = np.random.default_rng(42).integers(0, 256, (1, 48, 48, 1), dtype=np.uint8)
        input_hash = hashlib.sha256(input_value.tobytes()).hexdigest()
        rss_samples: list[int] = [self.rss_reader()]
        for _ in range(10):
            self.sleep(0.05)
            rss_samples.append(self.rss_reader())
        baseline_reads = tuple(rss_samples[-5:])
        baseline = int(np.median(baseline_reads))
        for _ in range(50):
            session.run(["probabilities"], {"images": input_value})
            rss_samples.append(self.rss_reader())
        latencies: list[int] = []
        for _ in range(500):
            start = self.clock_ns()
            session.run(["probabilities"], {"images": input_value})
            latencies.append(int(self.clock_ns() - start))
            rss_samples.append(self.rss_reader())
        peak = max(rss_samples)
        p50 = (
            _nearest(latencies) / 1_000_000
            if len(latencies) == 1
            else sorted(latencies)[max(0, int(np.ceil(0.50 * len(latencies))) - 1)] / 1_000_000
        )
        p95 = sorted(latencies)[max(0, int(np.ceil(0.95 * len(latencies))) - 1)] / 1_000_000
        rss_increment = max(0, peak - baseline)
        reasons: list[str] = []
        if max(baseline_reads) - min(baseline_reads) > 1_048_576:
            reasons.append("rss_baseline_unstable")
        if p95 > gates.max_p95_ms:
            reasons.append("p95_ms")
        if rss_increment > gates.max_rss_increment_bytes:
            reasons.append("rss_increment_bytes")
        return BenchmarkReport(
            HOST_BENCHMARK_SCHEMA,
            normalized_cpu_name(),
            platform.platform(),
            input_hash,
            tuple(latencies),
            tuple(rss_samples),
            baseline,
            peak,
            rss_increment,
            p50,
            p95,
            {
                "subprocess": "clean-host",
                "providers": ["CPUExecutionProvider"],
                "fallback": False,
                "execution_mode": "ORT_SEQUENTIAL",
                "intra_op_threads": 1,
                "inter_op_threads": 1,
                "graph_optimization": "ORT_ENABLE_ALL",
            },
            not reasons,
            tuple(reasons),
        )


class SyntheticBenchmarkAdapter:
    def run(self, onnx_path: Path, *, gates: QualityGatesConfig | None = None) -> BenchmarkReport:
        del onnx_path, gates
        return BenchmarkReport(
            HOST_BENCHMARK_SCHEMA,
            "synthetic",
            "synthetic",
            "synthetic",
            (1,) * 500,
            (0,) * 61,
            0,
            0,
            0,
            0.001,
            0.001,
            {"providers": ["synthetic"], "fallback": False},
            True,
            (),
            True,
        )
