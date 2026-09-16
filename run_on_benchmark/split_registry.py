"""Frozen case membership checks for thesis experiments."""

from __future__ import annotations

import hashlib
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

REGISTRY_PATH = Path(__file__).with_name("splits") / "thesis_split_v2.json"


@lru_cache(maxsize=1)
def load_split_registry() -> dict[str, Any]:
    value = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if value.get("schema_version") != 2:
        raise ValueError(f"unsupported split registry: {REGISTRY_PATH}")
    return value


def split_registry_sha256() -> str:
    return hashlib.sha256(REGISTRY_PATH.read_bytes()).hexdigest()


def _matches(case_id: str, member: str) -> bool:
    if member == case_id:
        return True
    match = re.fullmatch(r"([a-zA-Z_-]+):(\d+)-(\d+)", member)
    case_match = re.fullmatch(r"([a-zA-Z_-]+)-(\d+)", case_id)
    if not match or not case_match:
        return False
    prefix, start, end = match.groups()
    case_prefix, number = case_match.groups()
    return prefix.casefold() == case_prefix.casefold() and int(start) <= int(number) <= int(end)


def assigned_split(benchmark_id: str, case_id: str) -> str | None:
    benchmark = (load_split_registry().get("benchmarks") or {}).get(benchmark_id) or {}
    matches = [
        split for split, members in (benchmark.get("splits") or {}).items()
        if any(_matches(case_id, member) for member in members)
    ]
    if len(matches) > 1:
        raise ValueError(f"case belongs to multiple splits: {benchmark_id}/{case_id}: {matches}")
    return matches[0] if matches else None


def assert_case_split(benchmark_id: str, case_id: str, requested_split: str) -> str:
    actual = assigned_split(benchmark_id, case_id)
    if actual is None:
        raise ValueError(f"case is not assigned in frozen split registry: {benchmark_id}/{case_id}")
    if actual != requested_split:
        raise ValueError(
            f"split mismatch for {benchmark_id}/{case_id}: requested {requested_split}, frozen as {actual}"
        )
    return actual
