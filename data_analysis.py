"""
数据分析模块
根据考核评估数据 (DataFrame) 和地区名称，自动查找补充材料，
利用 LLM 生成分析查询指令，并通过 DataInspectorMCPTool 执行多轮数据分析。
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Union, Any

import pandas as pd

from llm import BaseLLM
from utils import logger
from utils.data_inspector import (
    describe_dataframes_schema,
    DataInspectorMCPTool,
)
from utils.file_io import read_all_excel
from utils.prompt_renderer import render_prompt


# 补充材料默认目录
_DETAILED_DATA_DIR = Path(__file__).parent / "data" / "detailed_data"


# ============================================================
# 主入口
# ============================================================

def analyze_region(
    assessment_df: pd.DataFrame,
    region_name: str,
    llm: BaseLLM,
    *,
    detailed_data_dir: Union[str, Path] = _DETAILED_DATA_DIR,
    supplementary_header=0,
    max_queries: int = 5,
    code_agent_model: Optional[str] = None,
    code_agent_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    """
    对指定地区进行完整的数据分析。

    流程:
        1. 在 detailed_data 目录下查找文件名包含地区名原文的 Excel 补充材料
        2. 使用 describe_dataframes_schema 获取考核数据 + 补充材料的表结构
        3. 将结构信息发送给 LLM，生成多条自然语言查询指令
        4. 通过 DataInspectorMCPTool 逐条执行查询
        5. 将所有查询结果拼接为完整字符串返回

    Args:
        assessment_df: 多地区多维考核指标 DataFrame
        region_name: 地区名称（如 "渝北区"），用于匹配补充材料文件名
        llm: BaseLLM 实例，用于生成分析查询指令（必须传入）
        detailed_data_dir: 补充材料所在目录，默认 data/detailed_data
        supplementary_header: 补充材料 Excel 的表头配置（同 read_all_excel 的 header 参数）
        max_queries: LLM 最多生成的查询指令数量，默认 5
        code_agent_model: 执行查询所用的 CodeAgent 模型（默认 None → 使用环境变量）
        code_agent_kwargs: 传递给 query_dataframes 的额外参数

    Returns:
        str: 所有查询结果拼接的完整分析字符串
    """
    detailed_data_dir = Path(detailed_data_dir)
    code_agent_kwargs = code_agent_kwargs or {}

    # ---- Step 1: 查找补充材料文件 ----
    supplementary_files = _find_supplementary_files(region_name, detailed_data_dir)
    logger.info(
        f"[{region_name}] 找到 {len(supplementary_files)} 个补充材料文件: "
        f"{[f.name for f in supplementary_files]}"
    )

    # ---- Step 2: 读取数据，生成 Schema ----
    # 考核评估数据
    assessment_dfs = {"考核评估数据": assessment_df}
    assessment_schema = describe_dataframes_schema(assessment_dfs)

    # 补充材料
    supplementary_dfs: Dict[str, pd.DataFrame] = {}
    for file_path in supplementary_files:
        try:
            file_dfs = read_all_excel(file_path, header=supplementary_header)
            file_name = file_path.stem
            for sheet_name, df in file_dfs.items():
                key = f"{file_name}__{sheet_name}"
                supplementary_dfs[key] = df
        except Exception as e:
            logger.warning(f"[{region_name}] 读取补充材料失败 {file_path.name}: {e}")

    supplementary_schema = (
        describe_dataframes_schema(supplementary_dfs)
        if supplementary_dfs
        else "（无补充材料）"
    )

    logger.info(
        f"[{region_name}] Schema 生成完成 — "
        f"考核数据: {assessment_df.shape}, "
        f"补充材料: {len(supplementary_dfs)} 个 Sheet"
    )

    # ---- Step 3: LLM 生成查询指令 ----
    query_instructions = _generate_query_instructions(
        llm=llm,
        region_name=region_name,
        assessment_schema=assessment_schema,
        supplementary_schema=supplementary_schema,
        max_queries=max_queries,
    )
    logger.info(
        f"[{region_name}] LLM 生成了 {len(query_instructions)} 条查询指令"
    )

    if not query_instructions:
        logger.warning(f"[{region_name}] LLM 未生成任何有效查询指令")
        return f"# {region_name} 数据分析报告\n\n未能生成有效的查询指令，请检查输入数据和 LLM 配置。"

    # ---- Step 4: 使用 MCP Tool 逐条执行查询 ----
    all_dfs = {**assessment_dfs, **supplementary_dfs}
    mcp_tool = DataInspectorMCPTool()
    results: List[str] = []

    for i, instruction in enumerate(query_instructions, 1):
        logger.info(
            f"[{region_name}] 执行查询 {i}/{len(query_instructions)}: "
            f"{instruction[:80]}..."
        )
        # 默认限制 CodeAgent 的最大步数，避免无限制多轮推理导致 token 暴涨
        effective_max_steps = code_agent_kwargs.get("max_steps", 4)

        # 其余传给 CodeAgent 的参数放在 agent_kwargs 中
        agent_kwargs = {
            k: v for k, v in code_agent_kwargs.items() if k != "max_steps"
        }

        result = mcp_tool.run({
            "action": "query",
            "dfs": all_dfs,
            "instruction": instruction,
            "model": code_agent_model,
            "max_steps": effective_max_steps,
            "agent_kwargs": agent_kwargs,
        })

        if "result" in result:
            results.append(
                f"### 查询 {i}: {instruction}\n\n{result['result']}"
            )
        else:
            error_msg = result.get("error", "未知错误")
            results.append(
                f"### 查询 {i}: {instruction}\n\n[查询失败] {error_msg}"
            )

        logger.info(f"[{region_name}] 查询 {i} 完成")

    # ---- Step 5: 汇总结果 ----
    final_result = (
        f"# {region_name} 数据分析报告\n\n"
        + "\n\n---\n\n".join(results)
    )
    logger.info(
        f"[{region_name}] 分析完成，共 {len(results)} 条查询结果，"
        f"总字符数: {len(final_result)}"
    )

    return final_result


# ============================================================
# 内部辅助函数
# ============================================================

def _find_supplementary_files(
    region_name: str,
    data_dir: Path,
) -> List[Path]:
    """
    在指定目录中查找文件名包含地区名称原文的 Excel 文件。

    Args:
        region_name: 地区名称（如 "渝北区"）
        data_dir: 搜索目录

    Returns:
        List[Path]: 匹配到的文件路径列表（按文件名排序）
    """
    if not data_dir.exists():
        logger.warning(f"补充材料目录不存在: {data_dir}")
        return []

    matched = [
        f
        for f in data_dir.iterdir()
        if f.is_file()
        and f.suffix.lower() in (".xlsx", ".xls")
        and region_name in f.stem
    ]

    return sorted(matched, key=lambda p: p.name)


def _generate_query_instructions(
    llm: BaseLLM,
    region_name: str,
    assessment_schema: str,
    supplementary_schema: str,
    max_queries: int = 5,
) -> List[str]:
    """
    利用 LLM 根据表结构信息，生成多条数据分析的自然语言查询指令。

    Args:
        llm: BaseLLM 实例
        region_name: 目标地区名称
        assessment_schema: 考核数据的结构描述
        supplementary_schema: 补充材料的结构描述
        max_queries: 最多生成的查询条数

    Returns:
        List[str]: 自然语言查询指令列表
    """
    system_prompt = render_prompt("data_analysis_system.j2")
    user_prompt = render_prompt(
        "data_analysis_user.j2",
        region_name=region_name,
        assessment_schema=assessment_schema,
        supplementary_schema=supplementary_schema,
        max_queries=max_queries,
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = llm.generate(messages)
    return _parse_query_instructions(response.content, max_queries)


def _parse_query_instructions(
    text: str,
    max_queries: int,
) -> List[str]:
    """
    从 LLM 输出中解析查询指令列表。
    支持:
      - 标准 JSON 数组
      - ```json ... ``` 代码块中的 JSON
      - 带 <think>...</think> 标签的输出（自动跳过思考链）
      - 回退：按行分割

    Args:
        text: LLM 原始输出
        max_queries: 最大条数上限

    Returns:
        List[str]: 解析后的查询指令列表
    """
    # 去除可能的 <think>...</think> 块（某些模型如 Qwen3 会输出思考链）
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # 尝试从 ```json ... ``` 代码块中提取
    json_match = re.search(
        r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL
    )
    json_str = json_match.group(1).strip() if json_match else cleaned

    # 尝试 JSON 解析
    try:
        instructions = json.loads(json_str)
        if isinstance(instructions, list):
            return [str(item).strip() for item in instructions if str(item).strip()][
                :max_queries
            ]
    except json.JSONDecodeError:
        pass

    # 如果整段文本中包含 JSON 数组，尝试提取
    array_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if array_match:
        try:
            instructions = json.loads(array_match.group(0))
            if isinstance(instructions, list):
                return [
                    str(item).strip() for item in instructions if str(item).strip()
                ][:max_queries]
        except json.JSONDecodeError:
            pass

    # 回退：按行分割，移除编号前缀
    logger.warning("无法解析 JSON 格式的查询指令，尝试按行分割")
    lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
    result = []
    for line in lines:
        # 移除编号前缀，如 "1. ", "1) ", "1、"
        line = re.sub(r"^\d+[\.\)、]\s*", "", line).strip()
        # 移除引号包裹
        line = line.strip('"').strip("'").strip()
        if line and not line.startswith(("{", "[", "```")):
            result.append(line)

    return result[:max_queries]
