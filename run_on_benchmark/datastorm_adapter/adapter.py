"""DataSTORM → InsightBench 适配器。

把 MyDataStorm 的 DataSTORMPipeline 包装成 InsightBench 兼容的 Agent 接口。

InsightBench 期望的接口：
    agent.get_insights(dataset_csv_path, ..., return_summary=True)
    → (pred_insights: list[str], pred_summary: str)

适配策略：
1. 用 CsvDatabaseBridge 替换 PostgreSQL DatabaseConnector
2. 禁用 Serper 网络搜索（InsightBench 是纯 CSV 场景）
3. 手动组装 DataSTORMPipeline 各组件，绕过其 __init__ 中的自动初始化
4. 从 FinalReport 中提取 insights list 和 summary
"""

from __future__ import annotations

import logging
import re
import sys
import os
from typing import Any

logger = logging.getLogger(__name__)

# 把 MyDataStorm 加入 Python 路径
# 目录结构:
#   D:\DataAgents\
#     MyDataStorm\               ← datastorm 包在这里
#     Report Generation\
#       run_on_benchmark\
#         datastorm_adapter\
#           adapter.py           ← 本文件
_ADAPTER_DIR    = os.path.dirname(os.path.abspath(__file__))   # .../datastorm_adapter
_RUN_ON_BENCH   = os.path.dirname(_ADAPTER_DIR)                # .../run_on_benchmark
_REPORT_GEN_DIR = os.path.dirname(_RUN_ON_BENCH)               # .../Report Generation
_DATAAGENTS_DIR = os.path.dirname(_REPORT_GEN_DIR)             # D:\DataAgents
_MYDATASTORM_DIR = os.path.join(_DATAAGENTS_DIR, "MyDataStorm")

for _path in [_MYDATASTORM_DIR, _RUN_ON_BENCH]:
    if _path not in sys.path:
        sys.path.insert(0, _path)

from datastorm.agents.executor import ExecutorAgent
from datastorm.agents.planner import PlannerAgent
from datastorm.config import (
    DataSTORMConfig,
    DatabaseConfig,
    ExplorationConfig,
    InternetConfig,
    LLMConfig,
    ReportConfig,
)
from datastorm.llm.client import LLMClient
from datastorm.internet.search import WebSearcher
from datastorm.modules.exploration import ExplorationFramework
from datastorm.modules.insight_bank import InsightBank
from datastorm.modules.report import ReportGenerator
from datastorm.modules.warmstart import WarmStartModule
from datastorm.pipeline import DataSTORMPipeline
from datastorm.types import FinalReport, Insight, Thesis

from datastorm_adapter.csv_db_bridge import CsvDatabaseBridge


class DataStormAdapter:
    """将 MyDataStorm 包装为 InsightBench 兼容的 Agent。

    参数：
        model_name:      LLM 模型名称（默认 gpt-4o）
        max_layers:      探索层数（默认 3，比论文的 5 少以节省 token）
        openai_api_key:  OpenAI API key（也可通过环境变量 OPENAI_API_KEY 设置）
        savedir:         结果保存目录（可选）
        verbose:         是否输出详细日志
    """

    def __init__(
        self,
        model_name: str = "gpt-5.4-mini",
        max_layers: int = 3,
        openai_api_key: str | None = None,
        savedir: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.model_name = model_name
        self.max_layers = max_layers
        self.savedir = savedir

        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        # 构造配置（不含 DB URL，DB 由 CsvDatabaseBridge 提供）
        self._base_config = DataSTORMConfig(
            llm=LLMConfig(
                api_key=openai_api_key or os.getenv("OPENAI_API_KEY", ""),
                exploration_model=model_name,
                report_model=model_name,
                temperature=0.7,
            ),
            database=DatabaseConfig(url="sqlite:///:memory:", database_type="SQLite"),
            internet=InternetConfig(serper_api_key=""),  # 禁用网络搜索
            exploration=ExplorationConfig(
                max_layers=max_layers,
                first_layer_max_questions=2,
                subsequent_layer_max_questions=3,
                executor_max_turns=10,
            ),
            report=ReportConfig(
                section_target_words=400,
                total_target_words=2000,
                max_web_queries_per_section=0,  # 禁用报告阶段的 web 查询
            ),
        )

    # ------------------------------------------------------------------
    # 主接口：与 InsightBench Agent.get_insights() 签名一致
    # ------------------------------------------------------------------

    def get_insights(
        self,
        dataset_csv_path: str,
        user_dataset_csv_path: str | None = None,
        goal: str = "Find interesting trends and patterns in this dataset",
        dataset_description: str = "",
        return_summary: bool = True,
    ) -> tuple[list[str], str] | list[str]:
        """在 CSV 数据集上运行 DataSTORM，返回 InsightBench 格式的结果。

        Args:
            dataset_csv_path:      主 CSV 文件路径
            user_dataset_csv_path: 可选的第二张表 CSV 路径
            goal:                  分析目标（来自 InsightBench metadata.goal）
            dataset_description:   数据集描述（来自 InsightBench metadata.dataset_description）
            return_summary:        是否同时返回 summary 字符串

        Returns:
            return_summary=True:  (pred_insights, pred_summary)
            return_summary=False: pred_insights
        """
        # 1. 构建查询字符串
        query = self._build_query(goal, dataset_description)
        logger.info("DataStormAdapter: query=%r", query[:200])

        # 2. 建立 CSV → SQLite 桥接
        bridge = CsvDatabaseBridge(
            csv_path=dataset_csv_path,
            table_name="main_table",
            user_csv_path=user_dataset_csv_path,
            user_table_name="user_table",
        )

        # 3. 手动组装 pipeline（注入 bridge 替代 PostgreSQL connector）
        config = self._base_config
        config.db_description = dataset_description

        pipeline = self._build_pipeline(config, bridge)

        try:
            report: FinalReport = pipeline.run(query)
        finally:
            bridge.close()

        # 4. 从 FinalReport 提取 insights 和 summary
        pred_insights = self._extract_insights(report)
        if return_summary:
            pred_summary = self._extract_summary(report)
            return pred_insights, pred_summary

        return pred_insights

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _build_query(self, goal: str, description: str) -> str:
        """把 InsightBench 的 goal + description 拼成 DataSTORM 的 query。"""
        parts = [goal]
        if description:
            parts.append(f"\nDataset context: {description[:500]}")
        return "\n".join(parts)

    def _build_pipeline(
        self, config: DataSTORMConfig, bridge: CsvDatabaseBridge
    ) -> DataSTORMPipeline:
        """手动组装 DataSTORMPipeline，用 bridge 替换 DatabaseConnector。

        绕过 DataSTORMPipeline.__init__ 中的 psycopg2 初始化，
        直接注入 CsvDatabaseBridge。
        """
        llm = LLMClient(config.llm)
        searcher = WebSearcher(config.internet)  # serper_api_key="" → 自动返回空结果

        planner = PlannerAgent(llm, config)
        executor = ExecutorAgent(llm, bridge, config)  # 注入 bridge

        insight_bank = InsightBank(llm, config)

        # 用 object.__new__ 跳过 __init__，再手动赋值各属性
        pipeline = object.__new__(DataSTORMPipeline)
        pipeline._config = config
        pipeline._llm = llm
        pipeline._db = bridge
        pipeline._searcher = searcher
        pipeline._planner = planner
        pipeline._executor = executor
        pipeline._insight_bank = insight_bank

        return pipeline

    def _extract_insights(self, report: FinalReport) -> list[str]:
        """从 FinalReport 中提取 insights list。

        优先从 references（每条引用对应一个数据库洞察），
        回退到从 markdown 正文中提取句子。
        """
        insights: list[str] = []

        # 方案 A：从 references 提取（每个 reference 是一个有据可查的洞察）
        if report.references:
            for ref in report.references:
                # reference 结构: {"id": ..., "question": ..., "answer": ..., "sql": ...}
                answer = ref.get("answer", "").strip()
                if answer and len(answer) > 20:
                    insights.append(answer)

        # 方案 B：如果 references 为空，从 markdown 中提取句子
        if not insights and report.markdown:
            insights = self._extract_sentences_from_markdown(report.markdown)

        return insights if insights else [report.thesis.title]

    def _extract_sentences_from_markdown(self, markdown: str) -> list[str]:
        """从 markdown 正文中提取有意义的句子作为 insights。"""
        lines = []
        for line in markdown.split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith(">"):
                continue
            line = re.sub(r"\[\^?\d+\]", "", line).strip()
            if len(line) > 40:
                lines.append(line)

        sentences: list[str] = []
        for line in lines:
            for sent in re.split(r"(?<=[.!?])\s+", line):
                sent = sent.strip()
                if len(sent) > 30:
                    sentences.append(sent)

        return sentences[:20]

    def _extract_summary(self, report: FinalReport) -> str:
        """从 FinalReport 提取 summary 字符串。"""
        summary_parts = [report.thesis.title]
        if report.thesis.research_strategy:
            summary_parts.append(report.thesis.research_strategy)
        if report.subtitle:
            summary_parts.append(report.subtitle)
        return " ".join(summary_parts)
