"""
主入口
从终端接收地区名，依次执行 数据分析 → 报告撰写 → 文本改写，
最终将报告保存到 output/ 目录。
"""

import sys
from pathlib import Path

from llm import OpenAILikeLLM, LLMConfig
from data_analysis import analyze_region
from doc_writing import DocWriter
from rewriting import Rewriter
from utils import logger
from utils.file_io import read_all_excel, data_save


# ============================================================
# 配置
# ============================================================

# 考核评估总表路径（按实际情况修改）
ASSESSMENT_FILE = Path("data/overview_data/考核评估总表.xlsx")
ASSESSMENT_HEADER = 0  # 表头配置，按实际情况修改

# 输出目录
OUTPUT_DIR = Path("output")


def _create_planning_llm() -> OpenAILikeLLM:
    """
    创建用于"数据分析规划"阶段的 LLM 客户端。
    此阶段使用默认模型（环境变量）即可。
    """
    return OpenAILikeLLM(config=LLMConfig())


def _create_writing_llm() -> OpenAILikeLLM:
    """
    创建用于"报告撰写"阶段的高级闭源 LLM 客户端。

    TODO: 替换为实际的闭源模型配置
    """
    return OpenAILikeLLM(config=LLMConfig(
        model="PLACEHOLDER_MODEL_NAME",         # TODO: 替换为实际模型名
        api_base="PLACEHOLDER_API_BASE",         # TODO: 替换为实际 API 地址
        api_key="PLACEHOLDER_API_KEY",           # TODO: 替换为实际 API Key
        temperature=0.7,
    ))


def _create_rewriting_llm() -> OpenAILikeLLM:
    """
    创建用于"文本改写/润色"阶段的高级闭源 LLM 客户端。

    TODO: 替换为实际的闭源模型配置
    """
    return OpenAILikeLLM(config=LLMConfig(
        model="PLACEHOLDER_MODEL_NAME",         # TODO: 替换为实际模型名
        api_base="PLACEHOLDER_API_BASE",         # TODO: 替换为实际 API 地址
        api_key="PLACEHOLDER_API_KEY",           # TODO: 替换为实际 API Key
        temperature=0.3,
    ))


# ============================================================
# 主流程
# ============================================================

def run(region_name: str) -> Path:
    """
    对指定地区执行完整的报告生成流程。

    Args:
        region_name: 地区名称（如 "渝北区"）

    Returns:
        Path: 最终报告保存路径
    """
    logger.info(f"===== 开始处理: {region_name} =====")

    # ---- 1. 读取考核评估总表 ----
    logger.info(f"[1/4] 读取考核评估数据: {ASSESSMENT_FILE}")
    dfs = read_all_excel(ASSESSMENT_FILE, header=ASSESSMENT_HEADER)
    # 取第一个 sheet（或按需调整）
    assessment_df = list(dfs.values())[0]
    logger.info(f"考核数据 shape: {assessment_df.shape}")

    # ---- 2. 数据分析 ----
    logger.info(f"[2/4] 数据分析: {region_name}")
    planning_llm = _create_planning_llm()
    analysis_result = analyze_region(
        assessment_df=assessment_df,
        region_name=region_name,
        llm=planning_llm,
    )
    logger.info(f"分析结果长度: {len(analysis_result)} 字符")

    # ---- 3. 报告撰写 ----
    logger.info(f"[3/4] 生成报告初稿: {region_name}")
    writing_llm = _create_writing_llm()
    writer = DocWriter(llm=writing_llm)
    draft = writer.write(
        analysis_result=analysis_result,
        assessment_df=assessment_df,
        region_name=region_name,
    )
    logger.info(f"初稿长度: {len(draft)} 字符")

    # ---- 4. 文本改写/润色 ----
    logger.info(f"[4/4] 改写润色: {region_name}")
    rewriting_llm = _create_rewriting_llm()
    rewriter = Rewriter(llm=rewriting_llm)
    final_report = rewriter.rewrite(draft)
    logger.info(f"最终报告长度: {len(final_report)} 字符")

    # ---- 5. 保存 ----
    output_path = data_save(
        data=final_report,
        file_path=OUTPUT_DIR / f"{region_name}_报告",
        file_type="md",
    )
    logger.info(f"报告已保存至: {output_path}")
    logger.info(f"===== 完成: {region_name} =====\n")

    return output_path


def main():
    if len(sys.argv) > 1:
        region_name = sys.argv[1]
    else:
        region_name = input("请输入地区名称: ").strip()

    if not region_name:
        print("错误: 地区名称不能为空")
        sys.exit(1)

    output_path = run(region_name)
    print(f"\n报告已生成: {output_path}")


if __name__ == "__main__":
    main()
