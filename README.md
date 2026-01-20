# 基于LLM的产业分析报告生成架构

# 架构设计

- Orchestrator, Agent, 最强模型驱动；负责：观察数据，提出假说，给出可数值验证的假设，调用SubAgents
- DataAnalysisAgent, 负责对收到的n个pandas dataframe, 生成执行代码（pandas, sklearn或者其他统计学相关的包?）并返回结果，考虑使用Smolagents/CodeAgent实现；
- SQLAgent, 负责处理原始的xlsx表格，基于Orchestrator的指令做join, groupby, select等操作。
- ReportWriter, 负责生成报告初稿
- StylePolisher, 负责保证报告的文风和用词可用

最后两个模块都应该可以在单次LLM调用下完成，主要任务是Prompt Engineering。暂时先不考虑。主要要做的是，顺利引导Orchestrator观察-思考给定的城区在各个指标上的表现情况，然后给出一系列可计算的统计学问题。SQL和DA两个模块如何使用CodeAgent完成，需要仔细考虑，其中细节还需要修改。