"""
evaluator.py — InsightBench / DACO 统一评估器

提供两种评估逻辑：
- InsightBench: LLM-as-Judge，One-to-Many Matching (0-1 分)
- DACO: Pairwise Comparison (胜率 %)
"""

import json
import random
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from llm import OpenAILikeLLM, LLMConfig
from utils import logger


# ============================================================
# InsightBench 评估
# ============================================================

def evaluate_insightbench(
    predictions_file: str,
    data_dir: str,
    evaluator_model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    评估 InsightBench 预测结果。

    Args:
        predictions_file: Agent 输出的 JSON 文件路径
        data_dir: benchmark 数据目录（用于加载 Ground-Truth）
        evaluator_model: 评估器模型名

    Returns:
        {
            "avg_insight_score": float,
            "avg_summary_score": float,
            "overall": float,
            "per_dataset": {dataset_name: {...}, ...},
        }
    """
    from run_on_benchmark.adapter_insightbench import load_ground_truth

    with open(predictions_file, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    evaluator = OpenAILikeLLM(config=LLMConfig(model=evaluator_model, temperature=0))

    data_dir = Path(data_dir)
    per_dataset = {}
    all_insight_scores = []
    all_summary_scores = []

    for ds_name, pred in predictions.items():
        # 查找 GT 目录
        gt_dir = data_dir / "data" / ds_name
        if not gt_dir.exists():
            gt_dir = data_dir / ds_name
        gt_json = data_dir / "data" / "notebooks" / f"{ds_name}.json"
        if not gt_dir.exists() and not gt_json.exists():
            logger.warning(f"[Eval] GT not found for {ds_name}, skipping")
            continue

        try:
            gt_target = gt_json if gt_json.exists() else gt_dir
            gt = load_ground_truth(str(gt_target))
        except FileNotFoundError:
            logger.warning(f"[Eval] Could not load GT for {ds_name}, skipping")
            continue

        gt_insights = gt.get("insights", [])
        gt_summary = gt.get("summary", "")
        pred_insights = [ins.get("insight", "") for ins in pred.get("insights", [])]
        pred_summary = pred.get("summary", "")

        # Insight 级别评分: One-to-Many
        insight_scores = []
        for gt_ins in gt_insights:
            gt_text = gt_ins if isinstance(gt_ins, str) else str(gt_ins)
            best = 0.0
            for pred_ins in pred_insights:
                score = _llm_eval_score(evaluator, gt_text, pred_ins)
                best = max(best, score)
            insight_scores.append(best)

        avg_ins = sum(insight_scores) / len(insight_scores) if insight_scores else 0.0

        # Summary 级别评分
        sum_score = _llm_eval_score(evaluator, gt_summary, pred_summary) if gt_summary else 0.0

        per_dataset[ds_name] = {
            "insight_score": round(avg_ins, 4),
            "summary_score": round(sum_score, 4),
            "n_gt_insights": len(gt_insights),
            "n_pred_insights": len(pred_insights),
        }
        all_insight_scores.append(avg_ins)
        all_summary_scores.append(sum_score)

    avg_insight = sum(all_insight_scores) / len(all_insight_scores) if all_insight_scores else 0.0
    avg_summary = sum(all_summary_scores) / len(all_summary_scores) if all_summary_scores else 0.0

    return {
        "avg_insight_score": round(avg_insight, 4),
        "avg_summary_score": round(avg_summary, 4),
        "overall": round((avg_insight + avg_summary) / 2, 4),
        "n_datasets_evaluated": len(per_dataset),
        "per_dataset": per_dataset,
    }


def _llm_eval_score(evaluator: OpenAILikeLLM, gt_text: str, pred_text: str) -> float:
    """用 LLM 评估两段文本的相似度，返回 0.0-1.0。"""
    if not gt_text.strip() or not pred_text.strip():
        return 0.0

    prompt = f"""You are evaluating a data analysis insight. Rate how well the Predicted text captures the key information in the Ground Truth.

Ground Truth:
{gt_text[:2000]}

Predicted:
{pred_text[:2000]}

Consider: (1) factual accuracy, (2) completeness, (3) specificity (numbers, trends).
Respond with ONLY a single number between 0.0 and 1.0 (e.g., "0.75"). Nothing else."""

    try:
        response = evaluator.chat(prompt)
        return _parse_score(response.content)
    except Exception as e:
        logger.warning(f"[Eval] LLM eval failed: {e}")
        return 0.0


def _parse_score(text: str) -> float:
    """从 LLM 响应中解析 0-1 分数。"""
    text = text.strip()
    # 尝试直接解析
    try:
        score = float(text)
        return max(0.0, min(1.0, score))
    except ValueError:
        pass
    # 尝试找第一个浮点数
    match = re.search(r"(\d+\.?\d*)", text)
    if match:
        score = float(match.group(1))
        return max(0.0, min(1.0, score))
    return 0.0


# ============================================================
# DACO 评估
# ============================================================

def evaluate_daco(
    predictions_file: str,
    ground_truth_file: str,
    evaluator_model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """
    评估 DACO 预测结果（Pairwise Comparison）。

    Args:
        predictions_file: Agent 输出的 JSON 文件
        ground_truth_file: GT 的 JSONL 文件
        evaluator_model: 评估器模型名

    Returns:
        {
            "average_helpfulness": float,  # 百分比，50 = 与 GT 持平
            "n_samples": int,
            "wins": int,
            "losses": int,
        }
    """
    with open(predictions_file, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    gt_data = {}
    with open(ground_truth_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            key = item["db_id"] + "|" + item["query"]
            gt_data[key] = item

    evaluator = OpenAILikeLLM(config=LLMConfig(model=evaluator_model, temperature=0))

    wins = 0
    total = 0

    for pred_item in predictions:
        key = pred_item["db_id"] + "|" + pred_item["query"]
        if key not in gt_data:
            logger.warning(f"[Eval] GT not found for {pred_item['db_id']}, skipping")
            continue

        gt_item = gt_data[key]
        pred_wins = _pairwise_compare(
            prediction=pred_item.get("prediction", {}),
            ground_truth=gt_item.get("answer", {}),
            query=gt_item["query"],
            db_title=gt_item["db_id"],
            evaluator=evaluator,
        )
        if pred_wins:
            wins += 1
        total += 1

    helpfulness = (wins / total * 100) if total > 0 else 0.0

    return {
        "average_helpfulness": round(helpfulness, 2),
        "n_samples": total,
        "wins": wins,
        "losses": total - wins,
    }


def _pairwise_compare(
    prediction: Dict,
    ground_truth: Dict,
    query: str,
    db_title: str,
    evaluator: OpenAILikeLLM,
) -> bool:
    """Pairwise comparison，返回 prediction 是否获胜。"""

    report_pred = _format_report(prediction)
    report_gt = _format_report(ground_truth)

    # 随机交换位置，避免位置偏差
    pred_is_1 = random.random() < 0.5
    if pred_is_1:
        report_1, report_2 = report_pred, report_gt
    else:
        report_1, report_2 = report_gt, report_pred

    prompt = f"""I have a database "{db_title}". {query}

I hired two analysts who produced these reports. Which is more helpful?

Evaluate by (in decreasing priority):
(1) Relevance to my analysis goal
(2) Insightfulness
(3) Diversity of perspectives

Answer with ONLY "Report-1" or "Report-2".

# Report-1
{report_1[:3000]}

# Report-2
{report_2[:3000]}

Answer:"""

    try:
        response = evaluator.chat(prompt)
        text = response.content.strip().lower()
        winner_is_1 = "report-1" in text or "report 1" in text
        return winner_is_1 == pred_is_1
    except Exception as e:
        logger.warning(f"[Eval] Pairwise compare failed: {e}")
        return False


def _format_report(report: Dict) -> str:
    """格式化 findings + suggestions。"""
    lines = []
    findings = report.get("findings", [])
    suggestions = report.get("suggestions", [])

    if findings:
        lines.append("## Findings")
        for i, f in enumerate(findings, 1):
            lines.append(f"{i}. {f}")

    if suggestions:
        lines.append("\n## Suggestions")
        for i, s in enumerate(suggestions, 1):
            lines.append(f"{i}. {s}")

    return "\n".join(lines) if lines else "(Empty report)"
