"""Dependency-free configuration boundary for the local DeepSeek scorer."""

from __future__ import annotations

import json
import os
from pathlib import Path

CANONICAL_DEEPSEEK_MODEL = "deepseek-flash"
DEFAULT_SCORER_ID = "local-deepseek-v41-thinking-v2"
DEPRECATED_ALIASES = {
    "deepseek-v4-flash": CANONICAL_DEEPSEEK_MODEL,
    "deepseek-v4-flash-vision-exp": CANONICAL_DEEPSEEK_MODEL,
    "deepseek-v4-pro": CANONICAL_DEEPSEEK_MODEL,
}


def canonicalize_model_name(model: str) -> str:
    return DEPRECATED_ALIASES.get(model, model)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except (OSError, ValueError, TypeError):
        return {}


def load_scorer_config(
    env: dict[str, str] | None = None,
    config_path: Path | None = None,
    legacy_path: Path | None = None,
) -> dict:
    """Load scorer settings independently while keeping credential compatibility.

    SCORER_* values always win.  The old MyDataStorm config remains a last-resort
    credential source so existing local setups do not break, but its business
    model never controls the judge model.
    """
    env = dict(os.environ if env is None else env)
    default_path = Path(__file__).with_name("scorer_config.json")
    selected = config_path or Path(env.get("SCORER_CONFIG", default_path))
    cfg = _read_json(selected)

    legacy = _read_json(legacy_path) if legacy_path else {}
    legacy_default = legacy.get("default") or legacy

    model = env.get("SCORER_MODEL") or cfg.get("model") or CANONICAL_DEEPSEEK_MODEL
    thinking = env.get("SCORER_THINKING") or cfg.get("thinking") or "enabled"
    if thinking not in {"enabled", "disabled"}:
        raise ValueError("SCORER_THINKING must be 'enabled' or 'disabled'")
    return {
        "api_key": (
            env.get("SCORER_API_KEY")
            or env.get("DEEPSEEK_API_KEY")
            or cfg.get("api_key")
            or legacy_default.get("api_key")
            or ""
        ),
        "api_base": (
            env.get("SCORER_API_BASE")
            or cfg.get("api_base")
            or legacy_default.get("api_base")
            or "https://api.deepseek.com"
        ),
        "model": canonicalize_model_name(model),
        "temperature": float(env.get("SCORER_TEMPERATURE") or cfg.get("temperature") or 0),
        "max_tokens": int(env.get("SCORER_MAX_TOKENS") or cfg.get("max_tokens") or 4096),
        "thinking": thinking,
    }
