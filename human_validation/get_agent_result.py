"""
交互式数据查询脚本
在终端中输入自然语言查询，由 Code Agent 执行并返回结果。
用法: python human_validation/get_agent_result.py
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd
import dotenv

# 确保项目根目录在 sys.path 中，以便导入项目模块
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

dotenv.load_dotenv(_PROJECT_ROOT / ".env")

from utils import logger
from utils.data_inspector import (
    describe_dataframes_schema,
    DataInspectorMCPTool,
)
from utils.file_io import read_all_excel

# ============================================================
# 变量设置区 — 按需修改
# ============================================================

# 地区名称
region_name = "江北区"

# 数据文件路径（支持绝对路径或相对于项目根目录的路径）
# 考核评估总表
assessment_file = _PROJECT_ROOT / "data" / "overview_data" / "考核评估总表.xlsx"
assessment_header = [0, 1, 2]  # 表头行配置

# 补充材料文件（可选，留空列表则只使用考核评估总表）
supplementary_files: List[str] = [
    # 示例: "data/detailed_data/江北区-25-06.xlsx",
]
supplementary_header = 0  # 补充材料表头行配置

# Code Agent 配置
code_agent_model = os.getenv("CODE_AGENT_MODEL_NAME")  # 从环境变量读取，或直接填写模型名
code_agent_kwargs: Dict[str, Any] = {}  # 额外参数，如 temperature, top_p 等
max_steps = 3  # Agent 最大执行步数


# ============================================================
# 数据加载
# ============================================================

def load_data() -> Dict[str, pd.DataFrame]:
    """加载所有配置的数据文件，返回 {sheet_name: DataFrame} 字典。"""
    all_dfs: Dict[str, pd.DataFrame] = {}

    # 读取考核评估总表
    if assessment_file.exists():
        try:
            assessment_dfs = read_all_excel(assessment_file, header=assessment_header)
            # 用 "考核评估数据" 作为 key（与 data_analysis.py 保持一致）
            for sheet_name, df in assessment_dfs.items():
                all_dfs[f"考核评估数据__{sheet_name}" if len(assessment_dfs) > 1 else "考核评估数据"] = df
            logger.info(f"已加载考核评估总表: {assessment_file.name}, {len(assessment_dfs)} 个 Sheet")
        except Exception as e:
            logger.warning(f"读取考核评估总表失败: {e}")
    else:
        logger.warning(f"考核评估总表不存在: {assessment_file}")

    # 读取补充材料
    for file_str in supplementary_files:
        file_path = Path(file_str) if Path(file_str).is_absolute() else _PROJECT_ROOT / file_str
        if not file_path.exists():
            logger.warning(f"补充材料文件不存在: {file_path}")
            continue
        try:
            file_dfs = read_all_excel(file_path, header=supplementary_header)
            file_name = file_path.stem
            for sheet_name, df in file_dfs.items():
                all_dfs[f"{file_name}__{sheet_name}"] = df
            logger.info(f"已加载补充材料: {file_path.name}, {len(file_dfs)} 个 Sheet")
        except Exception as e:
            logger.warning(f"读取补充材料失败 {file_path.name}: {e}")

    return all_dfs


def run_query(mcp_tool: DataInspectorMCPTool, dfs: Dict[str, pd.DataFrame], query_text: str) -> str:
    """执行单条自然语言查询，返回结果字符串。"""
    effective_max_steps = code_agent_kwargs.get("max_steps", max_steps)
    agent_kwargs = {k: v for k, v in code_agent_kwargs.items() if k != "max_steps"}

    result = mcp_tool.run({
        "action": "query",
        "dfs": dfs,
        "instruction": query_text,
        "model": code_agent_model,
        "max_steps": effective_max_steps,
        "agent_kwargs": agent_kwargs,
    })

    if "result" in result:
        return result["result"]
    else:
        return f"[查询失败] {result.get('error', '未知错误')}"


# ============================================================
# 主程序：交互式查询循环
# ============================================================

if __name__ == "__main__":
    print(f"=== 交互式数据查询 ({region_name}) ===")
    print("加载数据中...")

    all_dfs = load_data()

    if not all_dfs:
        print("错误: 没有加载到任何数据，请检查变量设置区的文件路径配置。")
        sys.exit(1)

    # 打印数据概览
    schema = describe_dataframes_schema(all_dfs)
    print(f"\n已加载 {len(all_dfs)} 个 Sheet:")
    for name, df in all_dfs.items():
        print(f"  - {name}: {df.shape}")
    print()

    mcp_tool = DataInspectorMCPTool()
    results: List[str] = []
    query_count = 0

    print('输入自然语言查询，按回车执行。输入 "quit" 或 "exit" 退出。')
    print("-" * 60)

    while True:
        try:
            query_text = input("\n查询> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if not query_text:
            continue
        if query_text.lower() in ("quit", "exit", "q"):
            break

        query_count += 1
        logger.info(f"[{region_name}] 执行查询 {query_count}: {query_text[:80]}...")

        result_text = run_query(mcp_tool, all_dfs, query_text)

        print(f"\n--- 查询 {query_count} 结果 ---")
        print(result_text)
        print("-" * 60)

        results.append(f"### 查询 {query_count}: {query_text}\n\n{result_text}")
        logger.info(f"[{region_name}] 查询 {query_count} 完成")

    # 汇总所有结果
    if results:
        final_report = f"# {region_name} 数据查询结果\n\n" + "\n\n---\n\n".join(results)
        print(f"\n=== 共完成 {len(results)} 条查询 ===")