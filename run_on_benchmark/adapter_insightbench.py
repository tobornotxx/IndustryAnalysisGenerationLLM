"""
adapter_insightbench.py — InsightBench 适配器

将我们的 Agent（CodeAgent + LLM）接入 InsightBench 的数据格式。
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
from code_agent import CodeAgent
from utils.data_inspector import describe_dataframes_schema
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
    dataset_dir = Path(dataset_dir)

    # ---- 1. 读入数据 ----
    csv_path = dataset_dir / "data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"data.csv not found in {dataset_dir}")

    df = pd.read_csv(csv_path)

    # 读取 goal
    goal = _read_goal(dataset_dir)

    # ---- 2. 构造 Schema ----
    schema = describe_dataframes_schema(
        {"data": df},
        max_sample_rows=3,
        max_unique_values=10,
    )

    # ---- 3. 生成分析问题 ----
    llm = OpenAILikeLLM(config=LLMConfig())
    questions = _generate_questions(llm, schema, goal, max_queries)
    logger.info(f"[InsightBench] 生成了 {len(questions)} 个分析问题")

    # ---- 4. 逐个问题执行代码分析 ----
    agent_kwargs = {}
    if code_agent_model:
        agent_kwargs["model"] = code_agent_model

    agent = CodeAgent(**agent_kwargs)
    insights: List[Dict[str, str]] = []

    for i, q in enumerate(questions, 1):
        logger.info(f"[InsightBench] 执行问题 {i}/{len(questions)}: {q['question'][:80]}")

        instruction = (
            f"你有一个 pandas DataFrame 变量 `df`，它来自 CSV 文件。\n"
            f"分析目标: {goal}\n"
            f"当前分析问题: {q['question']}\n\n"
            f"请编写 Python 代码分析数据，print 出关键的分析结果（数值、统计量、趋势描述等）。\n"
            f"不需要画图。用 print 输出结论性的文字和数值。"
        )

        result = agent.run(
            input=instruction,
            max_steps=code_agent_max_steps,
            additional_args={"df": df},
        )

        # 从代码执行结果中提取 insight
        insight_text = _extract_insight(llm, q["question"], result, goal)

        insights.append({
            "question": q["question"],
            "insight": insight_text,
            "type": q.get("type", "descriptive"),
        })

    # ---- 5. 生成 Summary ----
    summary = _generate_summary(llm, insights, goal)

    return {
        "insights": insights,
        "summary": summary,
    }


def load_ground_truth(dataset_dir: str) -> Dict[str, Any]:
    """
    从 InsightBench dataset 目录加载 Ground-Truth。

    尝试顺序：
    1. ground_truth.json（如果存在）
    2. 从 notebook.ipynb 中解析

    Returns:
        {"insights": [str, ...], "summary": str}
    """
    dataset_dir = Path(dataset_dir)

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


def _generate_questions(
    llm: OpenAILikeLLM,
    schema: str,
    goal: str,
    max_queries: int,
) -> List[Dict[str, str]]:
    """让 LLM 根据 Schema + Goal 生成多角度分析问题。"""

    prompt = f"""You are a data analytics expert. Given a dataset and an analysis goal, generate {max_queries} analysis questions.

Dataset Schema:
{schema}

Analysis Goal: {goal}

Generate exactly {max_queries} questions covering these types:
- descriptive: What happened? (distributions, summaries, counts)
- diagnostic: Why did it happen? (correlations, segmentation, root cause)
- predictive: What will likely happen? (trends, forecasting)
- prescriptive: What actions should be taken? (recommendations)

Return as JSON array:
[
  {{"question": "...", "type": "descriptive"}},
  {{"question": "...", "type": "diagnostic"}},
  ...
]

Return ONLY the JSON array, no other text."""

    response = llm.chat(prompt)
    return _parse_json_list(response.content, max_queries)


def _extract_insight(
    llm: OpenAILikeLLM,
    question: str,
    code_output: Optional[str],
    goal: str,
) -> str:
    """从代码执行结果中提取一句话 Insight。"""
    if not code_output:
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
            # 简单启发式：包含 "insight" / "finding" 的 markdown 单元格
            if any(kw in text.lower() for kw in ["insight", "finding", "observation"]):
                insights.append(text.strip())
            if any(kw in text.lower() for kw in ["summary", "conclusion", "recommendation"]):
                summary += text.strip() + "\n"

    return {
        "insights": insights,
        "summary": summary.strip(),
    }


def _parse_json_list(text: str, max_items: int) -> List[Dict[str, str]]:
    """从 LLM 输出中解析 JSON 列表。"""
    # 去掉 markdown 代码块
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    # 去掉 <think> 标签
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    try:
        items = json.loads(text.strip())
        if isinstance(items, list):
            return items[:max_items]
    except json.JSONDecodeError:
        pass

    # 回退：尝试找到第一个 [ 和最后一个 ]
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            items = json.loads(text[start : end + 1])
            if isinstance(items, list):
                return items[:max_items]
        except json.JSONDecodeError:
            pass

    # 最终回退：生成默认问题
    logger.warning("[InsightBench] 无法解析 LLM 返回的问题列表，使用默认问题")
    return [
        {"question": "What are the main distributions in this dataset?", "type": "descriptive"},
        {"question": "What trends or patterns exist over time?", "type": "diagnostic"},
        {"question": "What anomalies or outliers can be found?", "type": "diagnostic"},
    ]
