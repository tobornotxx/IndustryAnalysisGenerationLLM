"""
evaluator.py — InsightBench / DACO 评估器

使用 LLM-as-Judge 对预测结果与 Ground Truth 进行语义比对打分。
"""

import json
import re
from pathlib import Path
from typing import Dict, Any

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
    evaluator_model: str = "Pro/moonshotai/Kimi-K2.5",
) -> Dict[str, Any]:
    """
    评估 InsightBench 预测结果。

    Args:
        predictions_file: Agent 预测结果 JSON 文件路径
        data_dir: InsightBench 数据集根目录
        evaluator_model: 评估用 LLM 模型名

    Returns:
        dict: 包含 avg_insight_score, avg_summary_score, overall, per_dataset 等
    """
    from run_on_benchmark.adapter_insightbench import load_ground_truth

    with open(predictions_file, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    data_path = Path(data_dir)
    eval_llm = OpenAILikeLLM(config=LLMConfig(model=evaluator_model))

    per_dataset = {}
    total_insight_score = 0.0
    total_summary_score = 0.0
    n_evaluated = 0

    for ds_name, pred_data in predictions.items():
        # 加载 Ground Truth
        gt_path = data_path / "data" / "notebooks" / f"{ds_name}.json"
        if not gt_path.exists():
            gt_path = data_path / f"{ds_name}.json"
        if not gt_path.exists():
            gt_path = data_path / "data" / ds_name
        if not gt_path.exists():
            logger.warning(f"[Eval] GT not found for {ds_name}, skipping")
            continue

        try:
            gt = load_ground_truth(str(gt_path))
        except Exception as e:
            logger.warning(f"[Eval] Failed to load GT for {ds_name}: {e}")
            continue

        gt_insights = gt.get("insights", [])
        gt_summary = gt.get("summary", "")
        pred_insights = pred_data.get("insights", [])
        pred_summary = pred_data.get("summary", "")

        # 评估每条 insight
        insight_scores = []
        for gt_insight in gt_insights:
            gt_text = gt_insight if isinstance(gt_insight, str) else str(gt_insight)
            best_score = 0.0
            for pred_ins in pred_insights:
                pred_text = pred_ins.get("insight", "") if isinstance(pred_ins, dict) else str(pred_ins)
                score = _llm_judge_score(eval_llm, gt_text, pred_text, "insight")
                best_score = max(best_score, score)
            insight_scores.append(best_score)

        avg_insight = sum(insight_scores) / len(insight_scores) if insight_scores else 0.0

        # 评估 summary
        summary_score = 0.0
        if gt_summary and pred_summary:
            summary_score = _llm_judge_score(eval_llm, gt_summary, pred_summary, "summary")

        per_dataset[ds_name] = {
            "insight_score": round(avg_insight, 4),
            "summary_score": round(summary_score, 4),
            "n_gt_insights": len(gt_insights),
            "n_pred_insights": len(pred_insights),
        }

        total_insight_score += avg_insight
        total_summary_score += summary_score
        n_evaluated += 1

    avg_insight_score = total_insight_score / n_evaluated if n_evaluated > 0 else 0.0
    avg_summary_score = total_summary_score / n_evaluated if n_evaluated > 0 else 0.0

    return {
        "avg_insight_score": round(avg_insight_score, 4),
        "avg_summary_score": round(avg_summary_score, 4),
        "overall": round((avg_insight_score + avg_summary_score) / 2, 4),
        "n_datasets_evaluated": n_evaluated,
        "per_dataset": per_dataset,
    }


# ============================================================
# DACO 评估
# ============================================================

def evaluate_daco(
    predictions_file: str,
    ground_truth_file: str,
    evaluator_model: str = "Pro/moonshotai/Kimi-K2.5",
) -> Dict[str, Any]:
    """
    评估 DACO 预测结果。

    Args:
        predictions_file: Agent 预测结果 JSON 文件路径
        ground_truth_file: Ground Truth JSONL 文件路径
        evaluator_model: 评估用 LLM 模型名

    Returns:
        dict: 包含 average_helpfulness 等
    """
    with open(predictions_file, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    gt_items = []
    with open(ground_truth_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                gt_items.append(json.loads(line))

    eval_llm = OpenAILikeLLM(config=LLMConfig(model=evaluator_model))

    scores = []
    for pred_item in predictions:
        db_id = pred_item.get("db_id", "")
        query = pred_item.get("query", "")
        prediction = pred_item.get("prediction", {})

        # 找到对应的 GT
        gt_match = None
        for gt in gt_items:
            if gt.get("db_id") == db_id and gt.get("query") == query:
                gt_match = gt
                break

        if gt_match is None:
            continue

        pred_text = json.dumps(prediction, ensure_ascii=False)
        gt_text = json.dumps(gt_match, ensure_ascii=False)

        score = _llm_judge_score(eval_llm, gt_text, pred_text, "helpfulness")
        scores.append(score)

    avg_helpfulness = sum(scores) / len(scores) * 100 if scores else 0.0

    return {
        "average_helpfulness": round(avg_helpfulness, 1),
        "n_evaluated": len(scores),
    }


# ============================================================
# 内部辅助函数
# ============================================================

def _llm_judge_score(
    llm: OpenAILikeLLM,
    reference: str,
    prediction: str,
    eval_type: str = "insight",
) -> float:
    """
    使用 LLM 对预测与参考进行语义打分。

    Returns:
        float: 0.0 ~ 1.0 之间的分数
    """
    prompt = f"""You are an expert evaluator. Compare the prediction against the reference and rate the semantic similarity/quality.

Reference ({eval_type}):
{reference[:2000]}

Prediction ({eval_type}):
{prediction[:2000]}

Rate from 0.0 to 1.0 where:
- 1.0 = prediction captures the same key insight/finding as the reference
- 0.5 = prediction is partially relevant but misses important aspects
- 0.0 = prediction is irrelevant or completely wrong

Return ONLY a single number between 0.0 and 1.0, nothing else."""

    try:
        response = llm.chat(prompt)
        text = response.content.strip()
        match = re.search(r"(\d+\.?\d*)", text)
        if match:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))
        return 0.0
    except Exception as e:
        logger.warning(f"[Eval] LLM eval failed: {e}")
        return 0.0
"""
adapter_insightbench.py — InsightBench 适配器

将我们的 Agent（CodeAgent + LLM）接入 InsightBench 的数据格式。
复用 data_analysis.analyze_data() 作为核心分析引擎。
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd

import sys
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from llm import OpenAILikeLLM, LLMConfig
from data_analysis import analyze_data
from utils import logger


def run_agent_on_dataset(
    dataset_dir: str,
    max_queries: int = 5,
    code_agent_model: Optional[str] = None,
    code_agent_max_steps: int = 3,
) -> Dict[str, Any]:
    """
    在一个 InsightBench dataset 上运行 Agent。

    Args:
        dataset_dir: dataset 目录路径，内含 data.csv 和 goal.txt
        max_queries: 最多生成的分析问题数
        code_agent_model: CodeAgent 使用的模型（None=使用环境变量默认）
        code_agent_max_steps: 每个问题的最大代码执行步数

    Returns:
        {
            "insights": [{"question": str, "insight": str, "type": str}, ...],
            "summary": str,
        }
    """
    dataset_path = Path(dataset_dir)

    # ---- 1. 读入数据 ----
    df, goal = _load_dataset_and_goal(dataset_path)

    # ---- 2. 使用通用 analyze_data 执行分析 ----
    llm = OpenAILikeLLM(config=LLMConfig())

    query_results = analyze_data(
        dfs={"data": df},
        task_instruction=goal,
        llm=llm,
        max_queries=max_queries,
        schema_max_sample_rows=3,
        schema_max_unique_values=10,
        code_agent_model=code_agent_model,
        code_agent_kwargs={"max_steps": code_agent_max_steps},
    )

    logger.info(f"[InsightBench] analyze_data 返回 {len(query_results)} 条查询结果")

    # ---- 3. 从查询结果中提取 insights ----
    insights: List[Dict[str, str]] = []
    for i, qr in enumerate(query_results, 1):
        query_text = qr["query"]
        result_text = qr["result"]

        insight_text = _extract_insight(llm, query_text, result_text, goal)
        insights.append({
            "question": query_text,
            "insight": insight_text,
            "type": _infer_question_type(query_text),
        })

    # ---- 4. 生成 Summary ----
    summary = _generate_summary(llm, insights, goal)

    return {
        "insights": insights,
        "summary": summary,
    }


def _infer_question_type(question: str) -> str:
    """根据查询文本简单推断问题类型。"""
    q_lower = question.lower()
    if any(kw in q_lower for kw in ["预测", "趋势", "forecast", "predict", "trend"]):
        return "predictive"
    if any(kw in q_lower for kw in ["建议", "推荐", "应该", "recommend", "should", "action"]):
        return "prescriptive"
    if any(kw in q_lower for kw in ["为什么", "原因", "相关", "why", "cause", "correlat"]):
        return "diagnostic"
    return "descriptive"


def load_ground_truth(dataset_dir: str) -> Dict[str, Any]:
    """
    从 InsightBench dataset 目录加载 Ground-Truth。

    尝试顺序：
    1. ground_truth.json（如果存在）
    2. 从 notebook.ipynb 中解析

    Returns:
        {"insights": [str, ...], "summary": str}
    """
    dataset_path = Path(dataset_dir)

    # 支持直接传入新版 flag-*.json
    if dataset_path.is_file() and dataset_path.suffix.lower() == ".json":
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "insights": data.get("insights", []),
            "summary": data.get("summary", ""),
        }

    dataset_dir = dataset_path

    # 尝试预处理好的 JSON
    gt_file = dataset_dir / "ground_truth.json"
    if gt_file.exists():
        with open(gt_file, "r", encoding="utf-8") as f:
            return json.load(f)

    # 尝试从 notebook 解析
    notebook_file = dataset_dir / "notebook.ipynb"
    if notebook_file.exists():
        return _parse_notebook_gt(notebook_file)

    # 尝试 flag.json（InsightBench 的某些版本用这个名字）
    flag_file = dataset_dir / "flag.json"
    if flag_file.exists():
        with open(flag_file, "r", encoding="utf-8") as f:
            return json.load(f)

    raise FileNotFoundError(f"No ground-truth found in {dataset_dir}")


# ============================================================
# 内部辅助函数
# ============================================================

def _read_goal(dataset_dir: Path) -> str:
    """读取 goal 文本，支持多种文件名。"""
    for name in ["goal.txt", "goal.md", "metadata.json"]:
        p = dataset_dir / name
        if p.exists():
            if name.endswith(".json"):
                with open(p, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                return meta.get("goal", meta.get("description", ""))
            else:
                return p.read_text(encoding="utf-8").strip()
    return "Perform a comprehensive data analysis and find interesting insights."


def _load_dataset_and_goal(dataset_path: Path) -> tuple[pd.DataFrame, str]:
    """
    支持两种 InsightBench 数据格式:
    1) 旧格式目录: dataset_xxx/data.csv (+ goal.txt/metadata.json)
    2) 新格式文件: data/notebooks/flag-*.json (内含 dataset_csv_path 和 metadata.goal)
    """
    # 新版: 直接给 flag-*.json
    if dataset_path.is_file() and dataset_path.suffix.lower() == ".json":
        with open(dataset_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        csv_rel = meta.get("dataset_csv_path")
        if not csv_rel:
            raise FileNotFoundError(f"dataset_csv_path not found in {dataset_path}")

        csv_path = (dataset_path.parent.parent / csv_rel).resolve()
        if not csv_path.exists():
            # 兜底: 相对项目仓库根路径
            csv_path = (dataset_path.parents[2] / csv_rel).resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found for {dataset_path}: {csv_rel}")

        df = pd.read_csv(csv_path)
        goal = (
            meta.get("metadata", {}).get("goal")
            or meta.get("goal")
            or "Perform a comprehensive data analysis and find interesting insights."
        )
        return df, goal

    # 旧版: dataset 目录
    csv_path = dataset_path / "data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"data.csv not found in {dataset_path}")

    df = pd.read_csv(csv_path)
    goal = _read_goal(dataset_path)
    return df, goal


def _extract_insight(
    llm: OpenAILikeLLM,
    question: str,
    code_output: Optional[str],
    goal: str,
) -> str:
    """从代码执行结果中提取一句话 Insight。"""
    if not code_output or code_output.startswith("[查询失败]"):
        return "Analysis failed - no code output."

    prompt = f"""Based on the following code execution output, provide a concise one-paragraph insight.

Analysis Goal: {goal}
Question: {question}
Code Output:
{code_output[:3000]}

Write a clear, specific insight with concrete numbers/facts from the output. One paragraph only."""

    response = llm.chat(prompt)
    return response.content.strip()


def _generate_summary(
    llm: OpenAILikeLLM,
    insights: List[Dict[str, str]],
    goal: str,
) -> str:
    """汇总所有 Insight，生成最终 Summary。"""
    insights_text = "\n".join(
        f"- [{ins['type']}] {ins['insight']}" for ins in insights
    )

    prompt = f"""You are writing the executive summary of a data analysis report.

Analysis Goal: {goal}

All insights found:
{insights_text}

Write a concise summary (2-3 paragraphs) that:
1. Highlights the most important findings
2. Explains the key patterns and their implications
3. Provides 2-3 specific, actionable recommendations

Summary:"""

    response = llm.chat(prompt)
    return response.content.strip()


def _parse_notebook_gt(notebook_path: Path) -> Dict[str, Any]:
    """从 Jupyter Notebook 中粗略提取 GT insights 和 summary。"""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    insights = []
    summary = ""

    for cell in nb.get("cells", []):
        if cell["cell_type"] == "markdown":
            text = "".join(cell["source"])
            if any(kw in text.lower() for kw in ["insight", "finding", "observation"]):
                insights.append(text.strip())
            if any(kw in text.lower() for kw in ["summary", "conclusion", "recommendation"]):
                summary += text.strip() + "\n"

    return {
        "insights": insights,
        "summary": summary.strip(),
    }