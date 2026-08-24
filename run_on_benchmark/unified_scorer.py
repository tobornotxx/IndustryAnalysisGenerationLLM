"""统一打分器 —— 所有 benchmark 入口共用。

使用 G-Eval (LLM-as-Judge) 方法对 insight / summary 进行语义评分。
优先使用 logprobs 加权，API 不支持时自动回退到 Monte Carlo 采样。

配置来源：MyDataStorm/datastorm/llm_config.json
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)

# ============================================================
# 配置加载
# ============================================================

_CFG_PATH = (
    Path(__file__).resolve().parents[2] / "MyDataStorm" / "datastorm" / "llm_config.json"
)


def _load_config() -> dict:
    try:
        if _CFG_PATH.is_file():
            return json.loads(_CFG_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


_CFG = _load_config()
# llm_config.json 现为 {default: {...}, scenarios: {...}} 结构; 打分器取 default 块。
_DEFAULT_CFG = _CFG.get("default") or _CFG

_SCORER_API_KEY = _DEFAULT_CFG.get("api_key") or os.getenv("OPENAI_API_KEY", "")
_SCORER_API_BASE = _DEFAULT_CFG.get("api_base") or os.getenv("OPENAI_API_BASE", "")
_SCORER_MODEL = _DEFAULT_CFG.get("model_name") or _DEFAULT_CFG.get("model") or "deepseek-v4-pro"
_SCORER_TEMPERATURE = float(_DEFAULT_CFG.get("temperature", 0.7))
_SCORER_MAX_TOKENS = int(_DEFAULT_CFG.get("max_completion_tokens") or _DEFAULT_CFG.get("max_tokens") or 4096)

# Monte Carlo 采样次数
_MC_SAMPLES = 5
# logprobs 检测标志：None=未检测, True=支持, False=不支持
_logprobs_supported: bool | None = None

# ------------------------------------------------------------
# G-Eval prompt 变体开关
# ------------------------------------------------------------
# "gt_first"（默认）：Ground Truth 置于 Provided Answer 之前。
#   目的是让「system + 开头 + GT + instructions」成为稳定前缀，
#   使同一 GT 对多条 pred 的连续调用命中 prompt cache（外层循环为 GT，
#   内层遍历 pred，故同一前缀可复用 len(pred)-1 次）。
# "answer_first"：历史顺序（v8 及之前基线使用），用于 A/B 校验打分漂移。
# 可用环境变量 GEVAL_PROMPT_ORDER 覆盖，便于同一份代码跑对照实验。
_PROMPT_ORDER = os.getenv("GEVAL_PROMPT_ORDER", "gt_first")

# ------------------------------------------------------------
# token / 缓存用量累计（用于成本与缓存命中率核算）
# ------------------------------------------------------------
_USAGE = {
    "calls": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "cache_hit_tokens": 0,
    "cache_miss_tokens": 0,
}


def _record_usage(response: Any) -> None:
    """累计一次调用的 token 用量。

    DeepSeek 在 usage 中返回 prompt_cache_hit_tokens / prompt_cache_miss_tokens，
    OpenAI 则用 prompt_tokens_details.cached_tokens；两种都兼容，缺失则记 0。
    """
    try:
        u = getattr(response, "usage", None)
        if u is None:
            return
        _USAGE["calls"] += 1
        _USAGE["prompt_tokens"] += int(getattr(u, "prompt_tokens", 0) or 0)
        _USAGE["completion_tokens"] += int(getattr(u, "completion_tokens", 0) or 0)

        hit = getattr(u, "prompt_cache_hit_tokens", None)
        miss = getattr(u, "prompt_cache_miss_tokens", None)
        if hit is None:
            details = getattr(u, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", None) if details else None
            if cached is not None:
                hit = cached
                miss = int(getattr(u, "prompt_tokens", 0) or 0) - int(cached)
        _USAGE["cache_hit_tokens"] += int(hit or 0)
        _USAGE["cache_miss_tokens"] += int(miss or 0)
    except Exception:  # 记账失败绝不影响打分主流程
        pass


def get_usage_stats() -> dict[str, Any]:
    """返回打分器累计 token 用量 + 缓存命中率（供 result.json 落盘）。"""
    total_prompt = _USAGE["prompt_tokens"]
    hit = _USAGE["cache_hit_tokens"]
    return {
        **_USAGE,
        "prompt_order": _PROMPT_ORDER,
        "cache_hit_rate": round(hit / total_prompt, 4) if total_prompt else 0.0,
    }


def reset_usage_stats() -> None:
    """重置累计量（并行 worker 中每个 case 独立进程，一般无需调用）。"""
    for k in _USAGE:
        _USAGE[k] = 0


def _create_client() -> OpenAI:
    kwargs: dict[str, str] = {"api_key": _SCORER_API_KEY}
    if _SCORER_API_BASE:
        kwargs["base_url"] = _SCORER_API_BASE
    return OpenAI(**kwargs)


# ============================================================
# G-Eval Prompt
# ============================================================

_G_EVAL_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
    "Provided Answer:\n{answer}\n\n"
    "Ground Truth Answer:\n{gt_answer}\n\n"
    "Follow these instructions when writing your response:\n"
    "* On a scale of 1-10, provide a numerical rating for how close the provided answer "
    "is to the ground truth answer, with 10 denoting that the provided answer is the same "
    "as ground truth answer.\n"
    "* Your response should contain only the numerical rating. "
    "DONOT include anything else like the provided answer, the ground truth answer, "
    "or an explanation of your rating scale in your response.\n"
    "* Wrap your numerical rating inside <rating></rating> tags.\n"
    "* Check very carefully before answering.\n"
    "* Follow the output format as shown in the example below:\n"
    "Example response:\n<rating>7</rating>\n\n"
    "### Response:\n"
)

# GT 前置变体：把「开头 + Ground Truth + 全部固定 instructions」放在最前，
# 唯一变化的 {answer} 挪到末尾。这样在「同一 GT × 多条 pred」的内层循环中，
# 除首次外每次调用的前缀（约 400-500 token）都可命中 prompt cache。
# 语义与 _G_EVAL_TEMPLATE 完全一致，仅顺序不同——顺序是否影响打分需 A/B 验证。
_G_EVAL_TEMPLATE_GT_FIRST = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
    "Ground Truth Answer:\n{gt_answer}\n\n"
    "Follow these instructions when writing your response:\n"
    "* On a scale of 1-10, provide a numerical rating for how close the provided answer "
    "is to the ground truth answer, with 10 denoting that the provided answer is the same "
    "as ground truth answer.\n"
    "* Your response should contain only the numerical rating. "
    "DONOT include anything else like the provided answer, the ground truth answer, "
    "or an explanation of your rating scale in your response.\n"
    "* Wrap your numerical rating inside <rating></rating> tags.\n"
    "* Check very carefully before answering.\n"
    "* Follow the output format as shown in the example below:\n"
    "Example response:\n<rating>7</rating>\n\n"
    "Provided Answer:\n{answer}\n\n"
    "### Response:\n"
)


def _build_prompt(answer: str, gt_answer: str) -> str:
    """按 _PROMPT_ORDER 选择模板构建 G-Eval prompt。"""
    tpl = (
        _G_EVAL_TEMPLATE_GT_FIRST
        if _PROMPT_ORDER == "gt_first"
        else _G_EVAL_TEMPLATE
    )
    return tpl.format(answer=answer, gt_answer=gt_answer)

_SYSTEM_MESSAGE = (
    "You are a high school teacher evaluating student responses to a question. "
    "You are tasked with grading the response based on how well it answers the question. "
    "You are to provide a numerical rating for how well the response answers the question "
    "based on the ground truth answer."
)


# ============================================================
# logprobs 支持检测
# ============================================================

def _detect_logprobs(client: OpenAI, model: str) -> bool:
    """发送一次最小化 G-Eval 调用，检测 API 是否支持 logprobs 参数。"""
    prompt = _build_prompt(answer="test", gt_answer="test")
    try:
        client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_completion_tokens=20,
            logprobs=True,
            top_logprobs=3,
        )
        logger.info("Scorer: logprobs supported by %s", model)
        return True
    except Exception as e:
        msg = str(e)
        if any(kw in msg.lower() for kw in ("logprobs", "log_probs", "top_logprobs", "unsupported parameter")):
            logger.info("Scorer: logprobs NOT supported by %s, will use Monte Carlo", model)
            return False
        # 其他错误（网络、认证等）不判定为不支持
        logger.warning("Scorer: logprobs detection got unexpected error: %s", msg[:200])
        raise


# ============================================================
# 核心评分
# ============================================================

def _score_pair_logprobs(client: OpenAI, model: str, answer: str, gt_answer: str) -> float:
    """使用 logprobs 加权的 G-Eval 评分。"""
    prompt = _build_prompt(answer=answer, gt_answer=gt_answer)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        max_completion_tokens=50,
        logprobs=True,
        top_logprobs=5,
    )
    _record_usage(response)
    raw = response.choices[0].message.content or ""
    rating_match = re.findall(r"<rating>(\d+)</rating>", raw)
    if not rating_match:
        return _extract_fallback_rating(raw)

    rating_str = rating_match[0]
    logprobs_content = response.choices[0].logprobs.content
    if not logprobs_content:
        return float(rating_str) / 10.0

    tokens = [o.token for o in logprobs_content]
    if rating_str not in tokens:
        return float(rating_str) / 10.0

    idx = tokens.index(rating_str)
    top_lps = logprobs_content[idx].top_logprobs
    if not top_lps:
        return float(rating_str) / 10.0

    probs = [np.exp(lp.logprob) for lp in top_lps]
    probs = [p / sum(probs) for p in probs]
    ratings = [float(lp.token) if lp.token.isdigit() else 0 for lp in top_lps]
    score = sum(r * p for r, p in zip(ratings, probs))
    return score / 10.0


def _score_pair_monte_carlo(client: OpenAI, model: str, answer: str, gt_answer: str) -> float:
    """Monte Carlo 采样评分：多次调用取平均，作为 logprobs 的替代。"""
    prompt = _build_prompt(answer=answer, gt_answer=gt_answer)
    ratings: list[float] = []
    for _ in range(_MC_SAMPLES):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_completion_tokens=50,
            )
            _record_usage(response)
            raw = response.choices[0].message.content or ""
            rating_match = re.findall(r"<rating>(\d+)</rating>", raw)
            if rating_match:
                ratings.append(float(rating_match[0]))
        except Exception:
            continue
    if not ratings:
        return 0.0
    return sum(ratings) / len(ratings) / 10.0


def _extract_fallback_rating(raw: str) -> float:
    """从未包裹 <rating> 标签的响应中提取数值。"""
    nums = re.findall(r"\b(\d+)\b", raw)
    if nums:
        return float(nums[0]) / 10.0
    return 0.0


# ============================================================
# 公开 API
# ============================================================

def _resolve_score_func(client: OpenAI, model: str):
    """确定 logprobs 是否可用并返回对应的单对打分函数。"""
    global _logprobs_supported
    if _logprobs_supported is None:
        try:
            _logprobs_supported = _detect_logprobs(client, model)
        except Exception:
            logger.warning("Scorer: logprobs detection failed, using Monte Carlo")
            _logprobs_supported = False
    return _score_pair_logprobs if _logprobs_supported else _score_pair_monte_carlo


def score_insight_matrix(
    pred_insights: list[str], gt_insights: list[str]
) -> dict[str, Any]:
    """计算 pred × gt 打分矩阵，一次性给出 recall / precision / F1。

    这是 InsightEval(arXiv 2511.22884)定义的三个指标：
      recall    = E_gt  [ max_pred S(pred, gt) ]   —— 每条 GT 取最佳 pred（覆盖度）
      precision = E_pred[ max_gt   S(pred, gt) ]   —— 每条 pred 取最佳 GT（罚冗余）
      F1        = 2·R·P / (R+P)

    关键：三个指标共享同一个 len(gt) × len(pred) 矩阵，**不产生任何额外 API 调用**。
    历史上的 score_insights() 只用了「按行取 max」这一半信息。

    循环顺序为「外层 GT、内层 pred」：配合 gt_first prompt 模板，同一 GT 的前缀
    在内层连续调用中可命中 prompt cache（见 _G_EVAL_TEMPLATE_GT_FIRST）。
    """
    client = _create_client()
    model = _SCORER_MODEL
    score_func = _resolve_score_func(client, model)

    if not gt_insights or not pred_insights:
        return {
            "recall": 0.0, "precision": 0.0, "f1": 0.0,
            "matrix": [], "n_pairs": 0,
        }

    # matrix[i][j] = S(pred_j, gt_i)
    matrix: list[list[float]] = []
    for gt in gt_insights:                     # 外层 GT → 前缀稳定，利于缓存
        row = [score_func(client, model, pred, gt) for pred in pred_insights]
        matrix.append(row)

    arr = np.array(matrix, dtype=float)        # shape: (n_gt, n_pred)
    recall = float(arr.max(axis=1).mean())     # 每行(GT)取 max，再平均
    precision = float(arr.max(axis=0).mean())  # 每列(pred)取 max，再平均
    f1 = (
        2 * recall * precision / (recall + precision)
        if (recall + precision) > 0 else 0.0
    )

    return {
        "recall": recall,
        "precision": precision,
        "f1": float(f1),
        "matrix": matrix,
        "n_pairs": int(arr.size),
    }


def score_insights(pred_insights: list[str], gt_insights: list[str]) -> float:
    """对一组预测 insight 进行 G-Eval 评分（many-to-many best-match recall）。

    保留原签名与语义（纯 recall），确保 InsightBench 历史结果可比。
    需要 precision/F1 时改用 score_insight_matrix()。
    """
    return score_insight_matrix(pred_insights, gt_insights)["recall"]


def score_summary(pred_summary: str, gt_summary: str) -> float:
    """对单条 summary 进行 G-Eval 评分。"""
    client = _create_client()
    model = _SCORER_MODEL
    score_func = _resolve_score_func(client, model)
    return score_func(client, model, pred_summary, gt_summary)


def get_scorer_config() -> dict[str, str]:
    """返回当前打分器配置（用于日志/调试）。"""
    return {
        "api_base": _SCORER_API_BASE or "https://api.openai.com/v1",
        "model": _SCORER_MODEL,
        "logprobs": str(_logprobs_supported),
        "mc_samples": str(_MC_SAMPLES) if not _logprobs_supported else "N/A",
        "prompt_order": _PROMPT_ORDER,
    }
