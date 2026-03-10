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

    # ---- Step 2: 读取数据 ----
    # 考核评估数据
    assessment_dfs = {"考核评估数据": assessment_df}

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

    logger.info(
        f"[{region_name}] 数据读取完成 — "
        f"考核数据: {assessment_df.shape}, "
        f"补充材料: {len(supplementary_dfs)} 个 Sheet"
    )

    # ---- Step 3: LLM 生成查询指令 ----
    # 合并 schema 用于 LLM 规划（LLM 需要看到所有表的结构才能决定每条查询用哪些表）
    all_dfs = {**assessment_dfs, **supplementary_dfs}
    full_schema = describe_dataframes_schema(all_dfs)

    query_instructions = _generate_query_instructions(
        llm=llm,
        region_name=region_name,
        full_schema=full_schema,
        max_queries=max_queries,
    )
    logger.info(
        f"[{region_name}] LLM 生成了 {len(query_instructions)} 条查询指令"
    )

    if not query_instructions:
        logger.warning(f"[{region_name}] LLM 未生成任何有效查询指令")
        return f"# {region_name} 数据分析报告\n\n未能生成有效的查询指令，请检查输入数据和 LLM 配置。"

    # ---- Step 4: 使用 MCP Tool 逐条执行查询（只传入相关 Sheet）----
    mcp_tool = DataInspectorMCPTool()
    results: List[str] = []

    # 建立 sheet 名称到 df 的映射
    all_sheet_names = list(all_dfs.keys())

    for i, instr_item in enumerate(query_instructions, 1):
        query_text = instr_item["query"]
        requested_sheets = instr_item.get("sheets", [])

        logger.info(
            f"[{region_name}] 执行查询 {i}/{len(query_instructions)}: "
            f"{query_text[:80]}... | sheets={requested_sheets}"
        )

        # 筛选出本次查询需要的 DataFrame
        if requested_sheets:
            filtered_dfs = {}
            for sname in requested_sheets:
                if sname in all_dfs:
                    filtered_dfs[sname] = all_dfs[sname]
                else:
                    # 容错：sheet 名可能有细微差异，做模糊匹配
                    matched = [k for k in all_sheet_names if sname in k or k in sname]
                    if matched:
                        for m in matched:
                            filtered_dfs[m] = all_dfs[m]
                        logger.warning(
                            f"[{region_name}] Sheet '{sname}' 未精确匹配，"
                            f"模糊匹配到: {matched}"
                        )
                    else:
                        logger.warning(
                            f"[{region_name}] Sheet '{sname}' 不存在，跳过"
                        )
            # 如果筛选后为空，回退到全部 dfs
            if not filtered_dfs:
                logger.warning(
                    f"[{region_name}] 查询 {i} 的 sheets 全部无法匹配，"
                    f"回退使用全部数据"
                )
                filtered_dfs = all_dfs
        else:
            # 未指定 sheets，使用全部
            filtered_dfs = all_dfs

        logger.info(
            f"[{region_name}] 查询 {i} 实际使用 {len(filtered_dfs)} 个 Sheet: "
            f"{list(filtered_dfs.keys())}"
        )

        # 默认限制 CodeAgent 的最大步数
        effective_max_steps = code_agent_kwargs.get("max_steps", 3)
        agent_kwargs = {
            k: v for k, v in code_agent_kwargs.items() if k != "max_steps"
        }

        result = mcp_tool.run({
            "action": "query",
            "dfs": filtered_dfs,
            "instruction": query_text,
            "model": code_agent_model,
            "max_steps": effective_max_steps,
            "agent_kwargs": agent_kwargs,
        })

        if "result" in result:
            results.append(
                f"### 查询 {i}: {query_text}\n\n{result['result']}"
            )
        else:
            error_msg = result.get("error", "未知错误")
            results.append(
                f"### 查询 {i}: {query_text}\n\n[查询失败] {error_msg}"
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
    full_schema: str,
    max_queries: int = 5,
) -> List[Dict[str, Any]]:
    """
    利用 LLM 根据表结构信息，生成多条数据分析的查询指令（含涉及的 Sheet 名）。

    Args:
        llm: BaseLLM 实例
        region_name: 目标地区名称
        full_schema: 所有数据表的结构描述（合并后）
        max_queries: 最多生成的查询条数

    Returns:
        List[Dict]: 每项为 {"query": str, "sheets": List[str]}
    """
    system_prompt = render_prompt("data_analysis_system.j2")
    user_prompt = render_prompt(
        "data_analysis_user.j2",
        region_name=region_name,
        assessment_schema=full_schema,
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
) -> List[Dict[str, Any]]:
    """
    从 LLM 输出中解析查询指令列表。
    期望格式: [{"query": "...", "sheets": ["sheet1", ...]}, ...]
    
    支持:
      - 标准 JSON 数组
      - ```json ... ``` 代码块中的 JSON
      - 带 <think>...</think> 标签的输出（自动跳过思考链）
      - 回退：尝试旧格式（纯字符串数组），自动转换

    Args:
        text: LLM 原始输出
        max_queries: 最大条数上限

    Returns:
        List[Dict]: 每项为 {"query": str, "sheets": List[str]}
    """
    # 去除可能的 <think>...</think> 块
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # 尝试从 ```json ... ``` 代码块中提取
    json_match = re.search(
        r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL
    )
    json_str = json_match.group(1).strip() if json_match else cleaned

    # 尝试 JSON 解析
    parsed = _try_parse_json_array(json_str)
    if parsed is None:
        # 尝试从整段文本中提取 JSON 数组
        array_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if array_match:
            parsed = _try_parse_json_array(array_match.group(0))

    if parsed is not None:
        return _normalize_instructions(parsed, max_queries)

    # 回退：按行分割
    logger.warning("无法解析 JSON 格式的查询指令，尝试按行分割")
    lines = [line.strip() for line in cleaned.split("\n") if line.strip()]
    result = []
    for line in lines:
        line = re.sub(r"^\d+[\.\)、]\s*", "", line).strip()
        line = line.strip('"').strip("'").strip()
        if line and not line.startswith(("{", "[", "```")):
            result.append({"query": line, "sheets": []})
    return result[:max_queries]


def _try_parse_json_array(text: str) -> Optional[list]:
    """尝试将文本解析为 JSON 数组，失败返回 None。"""
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, TypeError):
        pass
    return None


def _normalize_instructions(
    raw_list: list,
    max_queries: int,
) -> List[Dict[str, Any]]:
    """
    将解析出的 JSON 数组标准化为 [{"query": str, "sheets": List[str]}] 格式。
    兼容旧格式（纯字符串数组）和新格式（dict 数组）。
    """
    result = []
    for item in raw_list:
        if isinstance(item, dict):
            query = str(item.get("query", "")).strip()
            sheets = item.get("sheets", [])
            if isinstance(sheets, str):
                sheets = [sheets]
            sheets = [s for s in sheets if isinstance(s, str) and s.strip()]
            if query:
                result.append({"query": query, "sheets": sheets})
        elif isinstance(item, str) and item.strip():
            # 兼容旧格式：纯字符串
            result.append({"query": item.strip(), "sheets": []})
    return result[:max_queries]
