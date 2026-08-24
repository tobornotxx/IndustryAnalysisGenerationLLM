"""CSV → SQLite 数据库桥接器。

实现与 datastorm.database.connector.DatabaseConnector 完全相同的接口，
但底层使用 SQLite + pandas，不依赖 PostgreSQL。

InsightBench 的数据是 CSV 文件，DataSTORM 的 ExecutorAgent 期望一个
DatabaseConnector 对象。这个类作为 drop-in 替换，让 ExecutorAgent
可以直接在 CSV 数据上运行 SQL 查询。
"""

from __future__ import annotations

import io
import logging
import sqlite3
import tempfile
import os
import sys
from typing import Any

import pandas as pd

# skill 包位于 MyDataStorm 内；bridge 在 IndustryAnalysisGenerationLLM 下，
# run_benchmark.py 通常已把 MyDataStorm 加进 sys.path，这里做路径兜底。
try:
    from datastorm.skills import get_skill_package, describe_categorical_columns
except ImportError:  # pragma: no cover
    _mds = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "MyDataStorm")
    )
    if os.path.isdir(_mds) and _mds not in sys.path:
        sys.path.insert(0, _mds)
    try:
        from datastorm.skills import get_skill_package, describe_categorical_columns
    except ImportError:
        get_skill_package = None  # type: ignore[assignment]
        describe_categorical_columns = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# 适合做「分组轴」的列的最大基数：超过则是 ID / 自由文本，分组无意义。
_MAX_CATEGORICAL_CARDINALITY = 50
# 分组轴的「理想基数」：用于给候选列排序（见 _pick_grouping_columns）。
# 纯结构参数，不针对任何数据集——基数太低（2-3，如布尔状态、系统元数据）
# 信息量不足，太高不便分组，中间段的业务维度列最有分析价值。
_IDEAL_CARDINALITY = 6
# 分类列 unique 值枚举上限（列序列化用）
_MAX_ENUM_VALUES = 12


def _period_alias(resample_rule: str) -> str:
    """把 resample 规则映射成 Series.dt.to_period 用的频率别名。

    to_period 不接受 'MS'/'QS'/'YS'/'h' 这类 offset 别名，需转成 'M'/'Q'/'A'/'H'。
    """
    return {
        "h": "H", "D": "D", "W": "W",
        "MS": "M", "QS": "Q", "YS": "Y",
    }.get(resample_rule, resample_rule)


class CsvDatabaseBridge:
    """将 CSV 文件暴露为可查询的 SQLite 数据库。

    接口与 datastorm.database.connector.DatabaseConnector 完全一致，
    可直接替换传入 ExecutorAgent。

    用法：
        bridge = CsvDatabaseBridge(
            csv_path="data.csv",
            table_name="incidents",
            user_csv_path="sysuser.csv",   # 可选第二张表
        )
        # 之后当作普通 DatabaseConnector 使用
        bridge.get_tables()
        bridge.execute_sql("SELECT * FROM incidents LIMIT 5")
    """

    def __init__(
        self,
        csv_path: str,
        table_name: str = "main_table",
        user_csv_path: str | None = None,
        user_table_name: str = "user_table",
    ) -> None:
        self._csv_path = csv_path
        self._table_name = table_name
        self._user_csv_path = user_csv_path
        self._user_table_name = user_table_name

        # 使用临时文件存储 SQLite DB（避免多进程冲突）
        self._db_fd, self._db_path = tempfile.mkstemp(suffix=".db")
        os.close(self._db_fd)

        self._conn: sqlite3.Connection | None = None
        self._table_descriptions: dict[str, str] = {}
        self._load_csvs()

    # ------------------------------------------------------------------
    # 内部：加载 CSV 到 SQLite
    # ------------------------------------------------------------------

    def _load_csvs(self) -> None:
        """把 CSV 文件加载进 SQLite 数据库。"""
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        # 适合做分组轴的分类列（跨所有表汇总），供 skill 引导渲染用
        self._grouping_columns: list[str] = []

        df = pd.read_csv(self._csv_path)
        df.to_sql(self._table_name, self._conn, if_exists="replace", index=False)
        self._table_descriptions[self._table_name] = (
            f"Main dataset loaded from {os.path.basename(self._csv_path)} "
            f"({len(df)} rows, {len(df.columns)} columns)"
        )
        self._schema_cache = self._build_schema_text(self._table_name, df)
        self._grouping_columns.extend(self._pick_grouping_columns(df))
        logger.info(
            "Loaded CSV '%s' → table '%s' (%d rows)",
            self._csv_path, self._table_name, len(df),
        )

        if self._user_csv_path and os.path.exists(self._user_csv_path):
            df_user = pd.read_csv(self._user_csv_path)
            df_user.to_sql(self._user_table_name, self._conn, if_exists="replace", index=False)
            self._table_descriptions[self._user_table_name] = (
                f"User dataset loaded from {os.path.basename(self._user_csv_path)} "
                f"({len(df_user)} rows, {len(df_user.columns)} columns)"
            )
            self._schema_cache += "\n\n" + self._build_schema_text(self._user_table_name, df_user)
            for c in self._pick_grouping_columns(df_user):
                if c not in self._grouping_columns:
                    self._grouping_columns.append(c)
            logger.info(
                "Loaded user CSV '%s' → table '%s' (%d rows)",
                self._user_csv_path, self._user_table_name, len(df_user),
            )

        logger.info(
            "Grouping columns for analytical guidance: %s",
            self._grouping_columns or "(none — will use generic wording)",
        )
        self._conn.commit()

    def _build_schema_text(self, table_name: str, df: pd.DataFrame) -> str:
        """生成可直接注入 prompt 的表结构描述（按列类型序列化）。

        序列化策略（吸收 DataGovBench arXiv 2607.06482 的消融结论）：
        该论文实测在真实政府数据上——
          · 只加 schema 几乎无增益（0.310 → 0.308）
          · 多塞样本行反而掉分（0.275），"LLM 在大池子里找不到关键信息"
          · 按列类型做紧凑统计摘要则大幅提升（+50 分）
        因此这里对每列按 dtype 给**针对性摘要**而非堆原始行：
          - 数值列   → min/max/mean + 四分位
          - 时间列   → range + 自适应粒度分布（原有 _temporal_profile）
          - 低基数列 → 枚举 unique 值（可直接当分组轴）
          - 高基数列 → 只给基数 + 少量样本（ID/自由文本，不适合分组）
        """
        lines = [f"Table: {table_name} ({len(df)} rows)"]
        lines.append("Columns:")
        for col in df.columns:
            s = df[col]
            dtype = str(s.dtype)
            n_unique = s.nunique()
            n_null = int(s.isnull().sum())
            lines.append(
                f"  - {col} ({dtype}, {n_unique} unique, {n_null} nulls)"
            )

            # 时间列：range + 自适应粒度分布
            temporal = self._temporal_profile(s)
            if temporal:
                for tline in temporal:
                    lines.append(f"      {tline}")
                continue

            # 数值列：分布摘要而非样本罗列
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                nn = s.dropna()
                if len(nn) > 0:
                    try:
                        q1, q2, q3 = nn.quantile([0.25, 0.5, 0.75])
                        lines.append(
                            f"      numeric: min={nn.min():g}, q1={q1:g}, median={q2:g}, "
                            f"q3={q3:g}, max={nn.max():g}, mean={nn.mean():g}"
                        )
                    except Exception:
                        lines.append(f"      numeric: min={nn.min():g}, max={nn.max():g}")
                continue

            # 低基数列：枚举取值 —— 这类列可直接作为分组轴
            if 0 < n_unique <= _MAX_ENUM_VALUES:
                vals = s.dropna().unique().tolist()
                lines.append(
                    "      values: [" + ", ".join(repr(v) for v in vals) + "]"
                )
            elif n_unique <= _MAX_CATEGORICAL_CARDINALITY:
                vals = s.dropna().unique()[:_MAX_ENUM_VALUES].tolist()
                lines.append(
                    f"      values ({n_unique} total, showing {len(vals)}): ["
                    + ", ".join(repr(v) for v in vals) + "]"
                )
            else:
                # 高基数（ID / 自由文本）：不适合分组，给少量样本即可
                vals = s.dropna().unique()[:3].tolist()
                lines.append(
                    f"      high-cardinality ({n_unique} distinct) — samples: ["
                    + ", ".join(repr(v) for v in vals) + "]"
                )
        return "\n".join(lines)

    @staticmethod
    def _pick_grouping_columns(df: pd.DataFrame, max_n: int = 6) -> list[str]:
        """挑出适合做分组轴的低基数分类列（数据无关的启发式）。

        规则：非数值、非时间、基数在 2.._MAX_CATEGORICAL_CARDINALITY 之间。

        排序不能用「基数升序」——那会让 2 值的状态列和 3 值的系统元数据列
        （如 updated_by = admin/system）挤掉基数 5 左右、分析价值高得多的
        业务维度列。改为按「离理想基数的距离」排序：基数太低信息量不足，
        太高不适合分组，中间段最有解释力。理想值 _IDEAL_CARDINALITY 是
        纯结构参数，不依赖任何数据集的具体列。

        这些列名会注入 skill 引导，取代 v8 里硬编码的 ServiceNow 字段名。
        """
        candidates: list[tuple[float, int, str]] = []
        for col in df.columns:
            s = df[col]
            if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
                continue
            if pd.api.types.is_datetime64_any_dtype(s):
                continue
            # object 列里的时间列（bridge 用文本存日期）也要排除
            if CsvDatabaseBridge._temporal_profile(s):
                continue
            n_unique = s.nunique()
            if 2 <= n_unique <= _MAX_CATEGORICAL_CARDINALITY:
                # 距理想基数越近越优先；同距时基数大的略优（信息量更足）
                candidates.append((abs(n_unique - _IDEAL_CARDINALITY), -n_unique, col))
        candidates.sort()
        return [c for _, _, c in candidates[:max_n]]

    @staticmethod
    def _temporal_profile(series: "pd.Series") -> list[str]:
        """若该列是时间列，返回 range + 自适应粒度分布的文本行；否则返回 []。

        粒度不写死：从数据实际跨度反推一个能让桶数落进 [8, 20] 的时间单位
        （小时/天/周/月/季/年），保证对任意跨度的时间序列都给出信息量适中、
        不爆炸的分布。这是通用 EDA profiling，不针对特定数据集假设。
        """
        import warnings

        s = series.dropna()
        if len(s) < 3:
            return []
        # 只对已是 datetime、或 object 且几乎全部可解析为日期的列启用
        if pd.api.types.is_datetime64_any_dtype(s):
            dt = s
        elif s.dtype == object:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dt = pd.to_datetime(s, errors="coerce")
            # 解析成功率不足 90% 则不认为是时间列（避免 ID / 纯数字误判）
            if dt.notna().mean() < 0.9:
                return []
            dt = dt.dropna()
        else:
            return []

        if len(dt) < 3:
            return []

        tmin, tmax = dt.min(), dt.max()
        span_days = (tmax - tmin).total_seconds() / 86400.0
        if span_days <= 0:
            return [f"time range: {tmin} → {tmax} (all values within a single instant)"]

        # —— 粒度自适应：目标 8~20 个桶 ——
        # (pandas resample rule, 单位名, 副词形式, 每单位天数) 从细到粗
        candidates = [
            ("h",  "hour",    "hourly",    1 / 24),
            ("D",  "day",     "daily",     1),
            ("W",  "week",    "weekly",    7),
            ("MS", "month",   "monthly",   30.44),
            ("QS", "quarter", "quarterly", 91.31),
            ("YS", "year",    "yearly",    365.25),
        ]
        target_max_buckets = 20
        rule, unit_name, adverb = candidates[-1][0], candidates[-1][1], candidates[-1][2]
        for r, name, adv, unit_days in candidates:
            if span_days / unit_days <= target_max_buckets:
                rule, unit_name, adverb = r, name, adv
                break

        counts = dt.dt.to_period(_period_alias(rule)).value_counts().sort_index()
        # 分布文本：紧凑单行 period:count；桶多时截断保护
        parts = [f"{str(p)}:{int(c)}" for p, c in counts.items()]
        dist_str = "  ".join(parts)
        peak_p = counts.idxmax()
        return [
            f"time range: {tmin} → {tmax} (span {span_days:.0f} days)",
            f"{adverb} counts: {dist_str}",
            f"(peak {unit_name}: {peak_p} with {int(counts.max())}; "
            f"mean {counts.mean():.1f}/{unit_name} — use this to pick an aggregation "
            f"scale with enough count per bucket before testing time trends)",
        ]


    # bridge 消费的 skill（顺序即注入顺序）。
    # v8 曾把这些引导硬编码在此处，且举例写了 category/assigned_to/priority/
    # assignment_group —— 那是 ServiceNow 工单字段，构成对 benchmark 的隐性
    # 过拟合（迁移到政府数据时 agent 会去找不存在的列）。现改为从 skill 包加载，
    # 列名在运行时由本数据集真实 schema 注入（见 _pick_grouping_columns）。
    _BRIDGE_SKILL_IDS = [
        "decompose-trend-by-category",
        "decompose-imbalance-by-group",
        "pick-adequate-time-bucket",
    ]

    def _build_analytical_guidance(self) -> str:
        """从 skill 包渲染分析引导，列名用本数据集的真实分类列填充。"""
        if get_skill_package is None or describe_categorical_columns is None:
            logger.warning("Skill package unavailable — schema context without guidance")
            return ""
        try:
            pkg = get_skill_package()
            return pkg.render_guidance(
                self._BRIDGE_SKILL_IDS,
                categorical_columns=describe_categorical_columns(self._grouping_columns),
            )
        except Exception as e:
            logger.warning("Skill guidance render failed (%s) — continuing without", e)
            return ""

    def get_schema_context(self) -> str:
        """返回完整的数据库 schema 上下文（用于注入 executor prompt）。"""
        return self._schema_cache + self._build_analytical_guidance()

    # ------------------------------------------------------------------
    # DatabaseConnector 兼容接口
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """兼容接口：SQLite 连接在 __init__ 中已建立。"""
        pass

    def close(self) -> None:
        """关闭 SQLite 连接并清理临时文件。"""
        if self._conn:
            self._conn.close()
            self._conn = None
        try:
            os.unlink(self._db_path)
        except OSError:
            pass

    def get_tables(self) -> str:
        """获取所有表及描述（对应 Executor action: get_tables）。"""
        lines = []
        for name, desc in self._table_descriptions.items():
            lines.append(f"- {name}: {desc}")
        return "\n".join(lines)

    def retrieve_tables_details(self, table_names: list[str]) -> str:
        """获取表的详细列信息（对应 Executor action: retrieve_tables_details）。"""
        assert self._conn is not None
        results = []
        for table_name in table_names:
            cursor = self._conn.execute(
                f"PRAGMA table_info('{table_name}')"
            )
            columns = cursor.fetchall()
            if not columns:
                results.append(f"Table '{table_name}': not found or no columns.")
                continue

            lines = [f"Table '{table_name}':"]
            for col in columns:
                # PRAGMA table_info: (cid, name, type, notnull, dflt_value, pk)
                nullable = "NOT NULL" if col[3] else "NULL"
                default = f" DEFAULT {col[4]}" if col[4] is not None else ""
                lines.append(f"  - {col[1]}: {col[2] or 'TEXT'} {nullable}{default}")

            count_row = self._conn.execute(
                f"SELECT COUNT(*) FROM \"{table_name}\""
            ).fetchone()
            row_count = count_row[0] if count_row else 0
            lines.append(f"  Row count: {row_count}")
            results.append("\n".join(lines))

        return "\n\n".join(results)

    def execute_sql(self, sql: str, max_rows: int = 50) -> tuple[pd.DataFrame, str]:
        """执行 SQL 查询（对应 Executor action: execute_sql）。"""
        assert self._conn is not None
        try:
            df = pd.read_sql_query(sql, self._conn)
        except Exception as e:
            empty = pd.DataFrame()
            return empty, f"SQL Error: {e}\n\nSQL: {sql}"

        total_rows = len(df)
        display_df = df.head(max_rows)
        omitted = total_rows - len(display_df)

        summary_parts = [f"Observed {total_rows} rows"]
        if omitted > 0:
            summary_parts.append(f"({omitted} omitted)")

        text = display_df.to_string(index=False) if not display_df.empty else "(empty result)"
        summary = ". ".join(summary_parts) + f".\n\nSQL: {sql}\nResult:\n{text}"
        return df, summary

    def execute_python_from_sql(self, sql: str, python_code: str) -> str:
        """基于 SQL 结果执行 Python 代码（对应 Executor action: execute_python_from_sql）。

        沙箱策略: 大胆放开。提供完整的 Python builtins, 预注入所有 prompt 中
        承诺的数据科学包 (numpy/pandas/scipy/sklearn/statsmodels/sympy/networkx/
        xgboost/lightgbm/polars/duckdb/lifelines/pingouin/ruptures/...),
        以及常用标准库模块 (re/collections/math/datetime/itertools/json/...)。
        LLM 写的任意 import 也照常放行。
        """
        df, _ = self.execute_sql(sql, max_rows=10000)

        output_buffer = io.StringIO()

        def captured_print(*args: Any, **kwargs: Any) -> None:
            kwargs["file"] = output_buffer
            print(*args, **kwargs)

        # 基础变量: SQL 结果 DataFrame
        local_vars: dict[str, Any] = {
            "sql_results": df,
            "print": captured_print,
        }

        # —— 预注入常用标准库 (LLM 经常直接引用而不 import) ——
        import re as _re, math, json as _json, itertools, datetime, collections, statistics
        from collections import Counter, defaultdict, OrderedDict
        local_vars.update({
            "re": _re, "math": math, "json": _json, "itertools": itertools,
            "datetime": datetime, "collections": collections, "statistics": statistics,
            "Counter": Counter, "defaultdict": defaultdict, "OrderedDict": OrderedDict,
        })

        # —— 预注入数据科学包 (与 executor prompt 中承诺的列表对齐) ——
        # 别名 → 模块名 (含子模块的用 attr 路径)
        def _try(alias: str, import_fn):
            try:
                local_vars[alias] = import_fn()
            except Exception:
                pass

        _try("np", lambda: __import__("numpy"))
        _try("numpy", lambda: __import__("numpy"))
        local_vars.setdefault("pd", pd)
        local_vars.setdefault("pandas", pd)
        _try("scipy", lambda: __import__("scipy"))
        _try("stats", lambda: __import__("scipy.stats", fromlist=["stats"]))
        _try("optimize", lambda: __import__("scipy.optimize", fromlist=["optimize"]))
        _try("sklearn", lambda: __import__("sklearn"))
        _try("statsmodels", lambda: __import__("statsmodels"))
        _try("sm", lambda: __import__("statsmodels.api", fromlist=["api"]))
        _try("smf", lambda: __import__("statsmodels.formula.api", fromlist=["api"]))
        for name in (
            "sympy", "networkx", "xgboost", "lightgbm", "polars", "duckdb",
            "lifelines", "pingouin", "ruptures", "category_encoders", "imblearn",
        ):
            _try(name, lambda n=name: __import__(n))

        # —— matplotlib: 强制无界面后端 Agg（必须在 import pyplot / seaborn 之前）——
        # 默认 GUI 后端在子进程/非主线程中绘图会告警甚至挂起。
        def _load_matplotlib():
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as _plt
            local_vars["plt"] = _plt
            return matplotlib
        _try("matplotlib", _load_matplotlib)
        # seaborn 在 matplotlib 后端锁定为 Agg 之后再导入
        _try("seaborn", lambda: __import__("seaborn"))

        # —— 大胆放开: 完整 builtins ——
        import builtins as _builtins
        full_builtins = dict(vars(_builtins))
        full_builtins["print"] = captured_print  # 重定向 print 到捕获缓冲

        try:
            exec(python_code, {"__builtins__": full_builtins}, local_vars)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            return (
                f"Python execution error: {type(e).__name__}: {e}\n\n"
                f"Traceback:\n{tb}\n\n"
                f"Available variables: sql_results (DataFrame with {len(df)} rows, columns: {list(df.columns)})\n"
                f"Pre-loaded: pd, np, scipy, stats, sklearn, statsmodels (sm), re, math, "
                f"Counter, collections, datetime, itertools, json — plus any package you import.\n"
                f"Tip: Fix the error and retry. Check column names match the DataFrame."
            )

        out = output_buffer.getvalue()
        if not out.strip():
            return (
                "(Python executed successfully but produced no output. "
                "Remember to print() the results you want to observe.)"
            )
        return out
