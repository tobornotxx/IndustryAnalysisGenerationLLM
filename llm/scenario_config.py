"""场景化 LLM 配置加载 (default + scenarios)。

配置文件位于项目根目录 ``llm_config.json``，结构与 MyDataStorm 一致::

    {
      "default":   { "model": "...", "api_base": "...", "api_key": "...", ... },
      "scenarios": {
        "planning":  { ... },   # 覆盖任意字段，未填的继承 default
        "code_agent": { ... },
        ...
      }
    }

每个场景可单独配置 model / api_base / api_key / temperature / top_p /
max_tokens / seed，未填写的字段自动继承 ``default``。

优先级 (由低到高):
    1. ``default`` 块
    2. ``scenarios.default`` 覆盖项
    3. ``scenarios[name]`` 覆盖项 (本场景)
    4. 环境变量 (MODEL_DEFAULT / API_BASE_DEFAULT / API_KEY_DEFAULT)  ← 仅作兜底
    5. 代码中显式传参  ← 最终覆盖 (在 LLMConfig / create_llm 层处理)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# 项目根目录下的 llm_config.json
_CFG_PATH = Path(__file__).resolve().parent.parent / "llm_config.json"

# 受支持的场景名 (未列出者仍可使用, 会回退到 default)
SUPPORTED_SCENARIOS: tuple[str, ...] = (
    "default",
    "planning",
    "code_agent",
    "writing",
    "rewriting",
)

# 场景可覆盖的字段 (同时兼容 model_name / model 两种写法)
_SCENARIO_FIELDS: tuple[str, ...] = (
    "model", "model_name",
    "api_base", "api_key",
    "temperature", "top_p",
    "max_tokens", "max_completion_tokens",
    "seed",
)


def _load_raw() -> dict:
    """加载 llm_config.json，文件不存在或解析失败返回空 dict。"""
    try:
        if _CFG_PATH.is_file():
            return json.loads(_CFG_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


_RAW: dict = _load_raw()
_DEFAULT_CFG: dict = _RAW.get("default") or {}
_SCENARIOS: dict = _RAW.get("scenarios") or {}


def _normalize(d: dict[str, Any]) -> dict[str, Any]:
    """把 model_name → model、max_completion_tokens → max_tokens 归一化。"""
    out = dict(d)
    if not out.get("model") and out.get("model_name"):
        out["model"] = out["model_name"]
    if "max_tokens" not in out and out.get("max_completion_tokens") is not None:
        out["max_tokens"] = out["max_completion_tokens"]
    return out


def resolve_scenario(name: str | None = None) -> dict[str, Any]:
    """返回某场景合并后的有效配置 (纯配置, 不读环境变量)。

    合并优先级: default 块 → scenarios.default → scenarios[name]。
    空字符串 / None 的覆盖项会被忽略, 以便只覆盖部分字段。
    """
    merged = _normalize(_DEFAULT_CFG)
    for src in ("default", name or "default"):
        override = _normalize(_SCENARIOS.get(src) or {})
        for key in _SCENARIO_FIELDS:
            val = override.get(key)
            if val not in (None, ""):
                merged[key] = val
    # 清理: 去掉归一化前的冗余键
    merged.pop("model_name", None)
    merged.pop("max_completion_tokens", None)
    return merged


def get_llm(scenario: str = "default", **overrides: Any):
    """按场景创建 OpenAILikeLLM 实例。

    Args:
        scenario: 场景名 (见 SUPPORTED_SCENARIOS)。
        **overrides: 显式覆盖字段, 优先级最高。

    Returns:
        OpenAILikeLLM 实例。
    """
    from llm.llm import LLMConfig, OpenAILikeLLM  # 延迟导入避免循环

    sc = resolve_scenario(scenario)
    # 环境变量仅作兜底 (配置文件优先)
    model = sc.get("model") or os.getenv("MODEL_DEFAULT", "")
    api_base = sc.get("api_base") or os.getenv("API_BASE_DEFAULT", "")
    api_key = sc.get("api_key") or os.getenv("API_KEY_DEFAULT", "")

    cfg = LLMConfig(
        model=overrides.pop("model", model),
        api_base=overrides.pop("api_base", api_base),
        api_key=overrides.pop("api_key", api_key),
        temperature=overrides.pop("temperature", sc.get("temperature", 0.7)),
        top_p=overrides.pop("top_p", sc.get("top_p", 1.0)),
        max_tokens=overrides.pop("max_tokens", sc.get("max_tokens")),
        seed=overrides.pop("seed", sc.get("seed")),
        **overrides,
    )
    return OpenAILikeLLM(config=cfg)
