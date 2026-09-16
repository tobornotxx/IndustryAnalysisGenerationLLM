"""Thin adapter around ServiceNow's official AgentPoirot implementation.

The upstream orchestration and prompts remain untouched. Only its GPT-only
transport functions are replaced with an OpenAI-compatible DeepSeek callable.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

CANONICAL_MODEL = "deepseek-flash"
UPSTREAM_DIR = Path(__file__).with_name("agent-poirot")


def upstream_commit(upstream_dir: Path = UPSTREAM_DIR) -> str:
    if not upstream_dir.exists():
        return "missing"
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=upstream_dir, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def make_deepseek_chat(
    *, api_key: str, api_base: str = "https://api.deepseek.com",
    model: str = CANONICAL_MODEL, client_factory: Callable[..., Any] | None = None,
) -> Callable[[str, str | None], str]:
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is required to run AgentPoirot")
    if client_factory is None:
        from openai import OpenAI
        client_factory = OpenAI
    client = client_factory(api_key=api_key, base_url=api_base)

    def chat(prompt: str, image: str | None = None) -> str:
        content: str | list[dict[str, Any]] = prompt
        if image and Path(image).is_file():
            image_path = Path(image)
            mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
            encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
            ]
        response = client.chat.completions.create(
            model=model, temperature=0,
            messages=[{"role": "user", "content": content}],
        )
        return response.choices[0].message.content or ""
    return chat


def install_transport(chat: Callable[[str, str | None], str], upstream_dir: Path = UPSTREAM_DIR):
    """Load official AgentPoirot and replace only its provider-specific calls."""
    if not (upstream_dir / "agentpoirot").is_dir():
        raise FileNotFoundError(
            f"AgentPoirot submodule missing at {upstream_dir}; run git submodule update --init"
        )
    if str(upstream_dir) not in sys.path:
        sys.path.insert(0, str(upstream_dir))
    import agentpoirot.agents.base as base_module
    import agentpoirot.agents.llms as llms_module
    import agentpoirot.agents.poirot as poirot_module

    factory = lambda _model_name, temperature=0: lambda prompt: chat(prompt, None)
    base_module.get_chat_model = factory
    poirot_module.get_chat_model = factory
    llms_module.prompt_llm = (
        lambda prompt, model=CANONICAL_MODEL, show_cost=False, image=None: chat(prompt, image)
    )
    return poirot_module.Poirot


def normalize_insights(items: list[Any]) -> list[str]:
    normalized = []
    for item in items:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            parts = [item.get("header"), item.get("question"), item.get("insight") or item.get("answer")]
            text = " ".join(str(part).strip() for part in parts if part).strip()
        else:
            text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def run_agentpoirot(
    *, csv_path: Path, goal: str, output_dir: Path, n_insights: int = 12,
    model: str = CANONICAL_MODEL, api_key: str | None = None,
    api_base: str = "https://api.deepseek.com",
) -> dict[str, Any]:
    import pandas as pd

    chat = make_deepseek_chat(
        api_key=api_key or os.getenv("DEEPSEEK_API_KEY", ""), api_base=api_base, model=model,
    )
    Poirot = install_transport(chat)
    output_dir.mkdir(parents=True, exist_ok=False)
    artifacts_dir = output_dir / "upstream_artifacts"
    artifacts_dir.mkdir()
    agent = Poirot(
        table=pd.read_csv(csv_path), savedir=str(artifacts_dir),
        model_name=model,
        meta_dict={"role": "data scientist", "goal": goal, "indicator_list": []},
    )
    items = agent.generate_insights(n_insights=n_insights, as_str=False)
    return {
        "schema_version": 1,
        "system_id": "agentpoirot-upstream-local",
        "system_source": "https://github.com/ServiceNow/agent-poirot",
        "system_commit": upstream_commit(),
        "generation_model": model,
        "goal": goal,
        "pred_insights": normalize_insights(items),
        "pred_summary": "",
        "summary_status": "not_exposed_by_upstream",
        "raw_output": items,
    }


def write_prediction(path: Path, prediction: dict[str, Any]) -> None:
    path.write_text(json.dumps(prediction, ensure_ascii=False, indent=2), encoding="utf-8")
