# Results 速查指引

## 目录总览

```
results/
├── README.md                 ← 本文件
├── datastorm/                ← DataSTORM benchmark 结果（由 run_benchmark.py 生成）
│   ├── run_info.json         │  本次运行的完整参数和配置
│   ├── summary.json          │  所有 flag 的汇总分数
│   └── flag-N/               │  每个数据集的详细结果
│       ├── result.json       │  预测 insights + GT + 打分
│       ├── datastorm_report.md  │  DataSTORM 生成的完整报告（叙事文本）
│       └── datastorm_report.json│  报告元数据（标题/论点/引用数）
│
├── report_agent/             ← Report Generation agent benchmark（由 run.py 生成）
│   ├── insightbench_predictions.json
│   └── insightbench_scores.json
│
└── regional_reports/         ← Report Generation 地区分析报告（由 main.py 生成）
    └── *_报告.md
```

---

## 各目录详解

### `datastorm/` — DataSTORM 在 InsightBench 上的 Benchmark

| 文件 | 内容 | 何时生成 |
|------|------|----------|
| `run_info.json` | 本次运行参数：benchmark_type, model_name, max_layers, scorer 配置 | 运行结束时 |
| `summary.json` | 所有 flag 的汇总：每个 flag 的 insight_score / summary_score / status | 运行中实时更新 |
| `flag-N/result.json` | 单条数据集详情：pred_insights (完整列表), gt_insights, scores, scorer 配置 | 每个数据集跑完后 |
| `flag-N/datastorm_report.md` | DataSTORM 生成的完整分析报告（markdown 格式，含 SQL 来源引用） | 每个数据集跑完后 |
| `flag-N/datastorm_report.json` | 报告元数据（标题、副标题、论点、引用数量） | 每个数据集跑完后 |

**运行命令**：
```bash
cd "Report Generation"
python run_on_benchmark/datastorm_adapter/run_benchmark.py --benchmark_type toy
python run_on_benchmark/datastorm_adapter/run_benchmark.py --benchmark_type full --n_datasets 5 --max_layers 3
```

---

### `report_agent/` — Report Generation Agent 在 InsightBench 上的结果

| 文件 | 内容 |
|------|------|
| `insightbench_predictions.json` | Agent 预测的 insights (question/insight/type) 和 summary |
| `insightbench_scores.json` | LLM-as-Judge 打分结果：avg_insight_score, avg_summary_score, overall |

**运行命令**：
```bash
cd "Report Generation"
python run_on_benchmark/run.py --benchmark insightbench --max_datasets 5
```

---

### `regional_reports/` — 地区分析报告

| 文件 | 内容 |
|------|------|
| `*_报告.md` | 各区最终生成的分析报告（含总分、亮点、不足、建议） |

**运行命令**：
```bash
cd "Report Generation"
python main.py
```

---

## 其他相关目录

| 路径 | 内容 |
|------|------|
| `logs/` | 按日期命名的应用运行日志（`2026-MM-DD.log`） |
| `benchmark_results/` | 历史 benchmark 结果和评估报告（旧版） |
| `benchmark_results/backups/` | 历史结果的备份 |
| `output/` | 旧的地区报告输出（已迁移到 `results/regional_reports/`） |
| `human_validation/` | 人工验证参考数据和 agent 查询结果 |
| `MyDataStorm/datastorm/llm_config.json` | LLM 统一配置（api_key / api_base / model） |
| `run_on_benchmark/insight-bench/data/notebooks/` | InsightBench 数据集定义（flag-N.json + GT） |
