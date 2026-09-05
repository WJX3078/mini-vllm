"""Lightweight metrics registry for the serving layer.

Counters + gauges + latency histograms, rendered as Prometheus text
format at /metrics. Deliberately dependency-free: a production deployment
can swap in prometheus_client without changing call sites.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque


class Metrics:
    def __init__(self, histogram_cap: int = 4096):
        self._counters: dict[str, int] = defaultdict(int)
        self._gauges: dict[str, float] = {}
        self._gauge_fns: dict[str, callable] = {}
        self._histograms: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=histogram_cap))
        self.started_at = time.time()

    # ---------------------------------------------------------------- write
    def inc(self, name: str, amount: int = 1):
        self._counters[name] += amount

    def set_gauge(self, name: str, value: float):
        self._gauges[name] = value

    def set_gauge_fn(self, name: str, fn):
        """Gauges computed at scrape time (scheduler/KV state)."""
        self._gauge_fns[name] = fn

    def observe(self, name: str, seconds: float):
        self._histograms[name].append(seconds)

    # ----------------------------------------------------------------- read
    def counter(self, name: str) -> int:
        return self._counters.get(name, 0)

    def histogram(self, name: str) -> list[float]:
        return list(self._histograms.get(name, ()))

    def percentile(self, name: str, p: float) -> float | None:
        xs = sorted(self.histogram(name))
        if not xs:
            return None
        return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]

    def render_prometheus(self) -> str:
        lines: list[str] = []
        for name, value in sorted(self._counters.items()):
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {value}")
        for name, value in sorted(self._gauges.items()):
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")
        for name, fn in sorted(self._gauge_fns.items()):
            try:
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {fn()}")
            except Exception:                     # engine state mid-change
                continue
        for name, hist in sorted(self._histograms.items()):
            lines.append(f"# TYPE {name} summary")
            xs = sorted(hist)
            if xs:
                for p in (50, 90, 95, 99):
                    v = xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]
                    lines.append(f'{name}{{quantile="{p / 100}"}} {v:.6f}')
                lines.append(f"{name}_sum {sum(xs):.6f}")
                lines.append(f"{name}_count {len(xs)}")
        lines.append("# TYPE minivllm_uptime_seconds gauge")
        lines.append(f"minivllm_uptime_seconds {time.time() - self.started_at:.1f}")
        return "\n".join(lines) + "\n"
