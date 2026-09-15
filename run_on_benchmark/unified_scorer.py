"""统一打分器 —— 所有 benchmark 入口共用。

使用 G-Eval (LLM-as-Judge) 方法对 insight / summary 进行语义评分。
优先使用 logprobs 加权，API 不支持时自动回退到 Monte Carlo 采样。

配置来源：MyDataStorm/datastorm/llm_config.json
"""

from __future__ import annotations

import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI
try:  # package import and legacy ``sys.path + import unified_scorer`` both work
    from .scorer_config import load_scorer_config
except ImportError:  # pragma: no cover - exercised by pi-agent's subprocess entrypoint
    from scorer_config import load_scorer_config

logger = logging.getLogger(__name__)

# ============================================================
# 配置加载
# ============================================================

_LEGACY_CFG_PATH = (
    Path(__file__).resolve().parents[2] / "MyDataStorm" / "datastorm" / "llm_config.json"
)


_SCORER_CFG = load_scorer_config(legacy_path=_LEGACY_CFG_PATH)
_SCORER_API_KEY = _SCORER_CFG["api_key"]
_SCORER_API_BASE = _SCORER_CFG["api_base"]
_SCORER_MODEL = _SCORER_CFG["model"]
_SCORER_TEMPERATURE = _SCORER_CFG["temperature"]
_SCORER_MAX_TOKENS = _SCORER_CFG["max_tokens"]

# Monte Carlo 采样次数
_MC_SAMPLES = 5
# 判分矩阵的并发线程数。矩阵内 len(gt)*len(pred) 次调用彼此独立，
# 串行会让单 case 判分耗时线性堆积（8*31=248 次串行需十几分钟）。
_SCORER_MAX_WORKERS = int(os.getenv("SCORER_MAX_WORKERS", "16"))
# logprobs 检测标志：None=未检测, True=支持, False=不支持
_logprobs_supported: bool | None = None
# 保证 logprobs 探测只发一次（并发时多线程会同时走到检测分支）
_DETECT_LOCK = threading.Lock()

# ------------------------------------------------------------
# G-Eval prompt 变体开关
# ------------------------------------------------------------
# 决定 prompt 中 Provided Answer / Ground Truth 的先后顺序，目的是让重复的
# 那一侧连同固定 instructions 构成稳定前缀以命中 prompt cache。
#   "pred_first"（默认）：pred 前置，配合外层 pred / 内层 GT 循环。
#       因实测 pred 比 GT 长约 20 倍，把 pred 当稳定前缀收益最大（~97% vs ~32%）。
#   "gt_first"    ：GT 前置（先前默认，实测仅 20.4% 命中，保留用于对照）。
#   "answer_first"：历史顺序（v8 及之前基线），用于 A/B 校验打分漂移。
# 可用环境变量 GEVAL_PROMPT_ORDER 覆盖，便于同一份代码跑对照实验。
_PROMPT_ORDER = os.getenv("GEVAL_PROMPT_ORDER", "pred_first")

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
# 判分矩阵并发执行，累加需加锁（+= 在多线程下非原子）
_USAGE_LOCK = threading.Lock()


def _record_usage(response: Any) -> None:
    """累计一次调用的 token 用量。

    DeepSeek 在 usage 中返回 prompt_cache_hit_tokens / prompt_cache_miss_tokens，
    OpenAI 则用 prompt_tokens_details.cached_tokens；两种都兼容，缺失则记 0。
    """
    try:
        u = getattr(response, "usage", None)
        if u is None:
            return
        prompt = int(getattr(u, "prompt_tokens", 0) or 0)
        completion = int(getattr(u, "completion_tokens", 0) or 0)

        hit = getattr(u, "prompt_cache_hit_tokens", None)
        miss = getattr(u, "prompt_cache_miss_tokens", None)
        if hit is None:
            details = getattr(u, "prompt_tokens_details", None)
            cached = getattr(details, "cached_tokens", None) if details else None
            if cached is not None:
                hit = cached
                miss = prompt - int(cached)

        with _USAGE_LOCK:
            _USAGE["calls"] += 1
            _USAGE["prompt_tokens"] += prompt
            _USAGE["completion_tokens"] += completion
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

# ── 缓存友好变体 ──
# 目标：让「重复出现的那一侧」连同全部固定 instructions 构成稳定前缀，
# 变化的一侧放末尾，从而在批量打分时命中 prompt cache。
#
# 实测数据（v9 flag-4）决定了该把哪一侧前置：
#   pred insight 中位数 ~2259 chars(~645 tok)
#   GT insight   中位数 ~106  chars(~30  tok)
# pred 比 GT 长约 20 倍。因此必须把**长的 pred** 当稳定前缀才有意义：
#   - gt_first   (GT 前置，外层 GT/内层 pred)：稳定前缀仅占 ~32% → 实测命中 20.4%
#   - pred_first (pred 前置，外层 pred/内层 GT)：稳定前缀占 ~97%
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

_G_EVAL_TEMPLATE_PRED_FIRST = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n"
    "Provided Answer:\n{answer}\n\n"
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
    "Ground Truth Answer:\n{gt_answer}\n\n"
    "### Response:\n"
)

_TEMPLATES = {
    "answer_first": _G_EVAL_TEMPLATE,          # v8 及之前的历史顺序（基线）
    "gt_first": _G_EVAL_TEMPLATE_GT_FIRST,
    "pred_first": _G_EVAL_TEMPLATE_PRED_FIRST,  # 默认：缓存收益最大
}


def _build_prompt(answer: str, gt_answer: str) -> str:
    """按 _PROMPT_ORDER 选择模板构建 G-Eval prompt。"""
    tpl = _TEMPLATES.get(_PROMPT_ORDER, _G_EVAL_TEMPLATE_PRED_FIRST)
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
    """确定 logprobs 是否可用并返回对应的单对打分函数。

    加锁保证探测只发一次：矩阵并发时多个线程会同时走到这里，
    无锁的话会重复发探测请求。
    """
    global _logprobs_supported
    with _DETECT_LOCK:
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

    并行化：矩阵里 len(gt)*len(pred) 次调用彼此独立，串行会让单 case 的判分
    耗时线性堆积（flag-4 是 8*31=248 次，串行需十几分钟）。这里用线程池并发，
    受 _SCORER_MAX_WORKERS 控制。

    循环顺序为「外层 pred、内层 GT」：配合 pred_first prompt 模板，同一条 pred 的
    前缀（含该 pred 全文 + 固定 instructions）在连续调用中可命中 prompt cache。
    之所以外层是 pred 而非 GT：实测 pred insight 中位数约 645 token，GT 仅约 30
    token，把长的那侧作为稳定前缀，缓存覆盖率从 ~32% 提升到 ~97%。
    """
    client = _create_client()
    model = _SCORER_MODEL
    score_func = _resolve_score_func(client, model)

    n_gt, n_pred = len(gt_insights), len(pred_insights)
    if not n_gt or not n_pred:
        return {
            "recall": 0.0, "precision": 0.0, "f1": 0.0,
            "matrix": [], "n_pairs": 0,
        }

    arr = np.zeros((n_gt, n_pred), dtype=float)

    def _one(i: int, j: int) -> tuple[int, int, float]:
        return i, j, score_func(client, model, pred_insights[j], gt_insights[i])

    # 任务顺序按 pred 分组（同一 pred 的 GT 连续提交），让缓存前缀更可能命中
    tasks = [(i, j) for j in range(n_pred) for i in range(n_gt)]
    workers = max(1, min(_SCORER_MAX_WORKERS, len(tasks)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_one, i, j) for i, j in tasks]
        for fut in as_completed(futures):
            try:
                i, j, s = fut.result()
                arr[i, j] = s
            except Exception as e:  # 单对失败记 0，不让整个 case 崩
                logger.warning("Scoring pair failed: %s", e)

    recall = float(arr.max(axis=1).mean())     # 每行(GT)取 max —— 覆盖度
    precision = float(arr.max(axis=0).mean())  # 每列(pred)取 max —— 罚冗余
    f1 = (
        2 * recall * precision / (recall + precision)
        if (recall + precision) > 0 else 0.0
    )

    return {
        "recall": recall,
        "precision": precision,
        "f1": float(f1),
        "matrix": arr.tolist(),
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
