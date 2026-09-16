"""Canonical experiment-system identities and claim boundaries."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

REGISTRY_PATH = Path(__file__).with_name("system_registry.json")


@lru_cache(maxsize=1)
def load_registry() -> dict[str, Any]:
    value = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or not isinstance(value.get("systems"), dict):
        raise ValueError(f"invalid system registry: {REGISTRY_PATH}")
    return value


def canonical_system_id(system_id: str, *, allow_retired: bool = False) -> str:
    registry = load_registry()
    if system_id in registry["systems"]:
        return system_id
    replacement = (registry.get("retired_aliases") or {}).get(system_id)
    if replacement and allow_retired:
        return str(replacement)
    if replacement:
        raise ValueError(f"retired system id {system_id!r}; use {replacement!r}")
    raise ValueError(f"unknown system id: {system_id!r}")


def system_record(system_id: str) -> dict[str, Any]:
    canonical = canonical_system_id(system_id)
    return dict(load_registry()["systems"][canonical])


def systems_with_generation_mode(mode: str) -> tuple[str, ...]:
    return tuple(
        system_id for system_id, record in load_registry()["systems"].items()
        if record.get("generation_mode") == mode
    )
