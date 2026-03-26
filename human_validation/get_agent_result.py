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

# 变量设置区
region_name = '江北区'
all_dfs = pd.DataFrame([])
query_instructions = []
code_agent_kwargs = {}
code_agent_model = None

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