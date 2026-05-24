"""Append-only JSONL usage log.

Best-effort: every failure is swallowed so logging never breaks a request.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def default_path() -> Path:
    custom = os.environ.get("USAGE_LOG_PATH", "").strip()
    if custom:
        return Path(custom).expanduser()
    return Path.home() / ".ghostpilot" / "usage.jsonl"


def log_usage(event: dict, *, path: Optional[Path] = None, enabled: bool = True) -> bool:
    """Append `event` as a single JSON line. Returns True on success."""
    if not enabled:
        return False
    target = path or default_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        record = {"ts": time.time(), **event}
        with target.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("usage_log write failed: %s", e)
        return False


def read_usage(path: Optional[Path] = None) -> list[dict]:
    """Read all logged events; empty list if missing/invalid."""
    target = path or default_path()
    if not target.exists():
        return []
    out: list[dict] = []
    try:
        with target.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except Exception as e:  # noqa: BLE001
        logger.warning("usage_log read failed: %s", e)
    return out


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return float(s[k])


def summarize(events: list[dict]) -> dict:
    """Aggregate counts + token totals + latency percentiles by provider."""
    total_in = 0
    total_out = 0
    latencies: list[float] = []
    by_provider: dict[str, dict] = {}
    for e in events:
        in_t = int(e.get("in") or 0)
        out_t = int(e.get("out") or 0)
        total_in += in_t
        total_out += out_t
        ms = e.get("total_ms")
        if isinstance(ms, (int, float)):
            latencies.append(float(ms))
        prov = str(e.get("provider") or "?")
        bp = by_provider.setdefault(prov, {"n": 0, "in": 0, "out": 0, "ms": []})
        bp["n"] += 1
        bp["in"] += in_t
        bp["out"] += out_t
        if isinstance(ms, (int, float)):
            bp["ms"].append(float(ms))
    for prov, bp in by_provider.items():
        ms = bp.pop("ms")
        bp["p50_ms"] = int(_percentile(ms, 50))
        bp["p95_ms"] = int(_percentile(ms, 95))
    return {
        "n": len(events),
        "total_in": total_in,
        "total_out": total_out,
        "p50_ms": int(_percentile(latencies, 50)),
        "p95_ms": int(_percentile(latencies, 95)),
        "by_provider": by_provider,
    }
