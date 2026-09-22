"""常驻 Python worker —— 通过行分隔 JSON 协议为 pi harness 提供数据分析能力。

设计意图（见 run_on_benchmark/PI_MIGRATION_PLAN.md §3）：
pi harness (TypeScript) 拥有 agent loop / 工具调度 / skill 加载；Python 只做
数据分析执行——那些库（pandas/sklearn/statsmodels/ruptures…）没有 JS 等价物。

为什么是常驻进程而不是每次 spawn：
    实测 import 全部数据科学库需约 1.2 秒。若每次 tool 调用都新起进程，
    单 case 数百次调用就要多付几分钟纯启动开销。常驻后这 1.2 秒只付一次，
    且 CSV 载入的 SQLite 连接可跨调用复用。

协议：stdin 每行一个 JSON 请求，stdout 每行一个 JSON 响应。
    {"id": "...", "op": "load_csv", "args": {...}}
    → {"id": "...", "ok": true, "result": {...}}
    → {"id": "...", "ok": false, "error": "..."}

关键约束：所有 print 必须走 stderr 或被捕获，绝不能污染 stdout ——
stdout 是协议通道。

复用而非重写：schema 生成、字符预算、时间画像全部直接调用现有的
CsvDatabaseBridge，避免两份实现漂移。
"""

from __future__ import annotations

import json
import os
import sys
import traceback

# 现有 bridge 在 run_on_benchmark/datastorm_adapter 下；MyDataStorm 提供 skill 包
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))           # IndustryAnalysisGenerationLLM
_RUN_ON_BENCH = os.path.join(_REPO, "run_on_benchmark")
_MYDATASTORM = os.path.join(os.path.dirname(_REPO), "MyDataStorm")
for _p in (_RUN_ON_BENCH, _MYDATASTORM):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from datastorm_adapter.csv_db_bridge import CsvDatabaseBridge  # noqa: E402


def _repair_surrogateescaped_text(value: str) -> str:
    """Recover legacy Windows bytes decoded through Python's surrogateescape."""
    repaired: list[str] = []
    for char in value:
        codepoint = ord(char)
        if 0xDC80 <= codepoint <= 0xDCFF:
            try:
                repaired.append(bytes([codepoint - 0xDC00]).decode("cp1252"))
            except UnicodeDecodeError:
                repaired.append("\ufffd")
        elif 0xD800 <= codepoint <= 0xDFFF:
            repaired.append("\ufffd")
        else:
            repaired.append(char)
    return "".join(repaired)


def _sanitize_unicode(value):
    if isinstance(value, str):
        return _repair_surrogateescaped_text(value)
    if isinstance(value, dict):
        return {
            _repair_surrogateescaped_text(str(key)): _sanitize_unicode(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_unicode(item) for item in value]
    return value


class Worker:
    """持有一个 CsvDatabaseBridge，处理来自 harness 的操作请求。"""

    def __init__(self) -> None:
        self._bridge: CsvDatabaseBridge | None = None

    # ------------------------------------------------------------------
    # 操作
    # ------------------------------------------------------------------

    def op_load_csv(self, csv_path: str, table_name: str = "main_table",
                    user_csv_path: str | None = None,
                    db_path: str | None = None,
                    reuse_db: bool = False) -> dict:
        """载入 CSV 到 SQLite，返回可注入 prompt 的 schema 上下文。

        共享 DB（内存优化）：池里第一个 worker 正常建库并返回 db_path；
        其余 worker 用 (db_path=<那个路径>, reuse_db=True) 只连接不重写数据。
        这样 N 个 worker 只存一份表数据，而不是各持一份完整副本。
        """
        if self._bridge is not None:
            self._bridge.close()
        self._bridge = CsvDatabaseBridge(
            csv_path=csv_path,
            table_name=table_name,
            user_csv_path=user_csv_path,
            db_path=db_path,
            reuse_db=reuse_db,
        )
        return {
            "schema_context": self._bridge.get_schema_context(),
            "tables": self._bridge.get_tables(),
            "grouping_columns": list(getattr(self._bridge, "_grouping_columns", [])),
            "db_path": self._bridge.db_path,
        }

    def op_sql(self, sql: str, max_rows: int = 50) -> dict:
        """执行 SQL，返回 executor 可读的文本摘要 + 行数。"""
        sql = _repair_surrogateescaped_text(sql)
        _, summary = self._require_bridge().execute_sql(sql, max_rows=max_rows)
        return {"summary": summary}

    def op_python(self, sql: str, code: str) -> dict:
        """先执行 SQL 取出 DataFrame（变量名 sql_results），再跑 Python 代码。

        沙箱已预注入 numpy/pandas/scipy/sklearn/statsmodels/duckdb/ruptures 等，
        见 CsvDatabaseBridge.execute_python_from_sql。
        """
        sql = _repair_surrogateescaped_text(sql)
        code = _repair_surrogateescaped_text(code)
        return {"output": self._require_bridge().execute_python_from_sql(sql, code)}

    def op_tables(self) -> dict:
        return {"tables": self._require_bridge().get_tables()}

    def op_table_details(self, table_names: list[str]) -> dict:
        return {"details": self._require_bridge().retrieve_tables_details(table_names)}

    def op_ping(self) -> dict:
        """健康检查，同时报告是否已载入数据。"""
        return {"pong": True, "loaded": self._bridge is not None, "pid": os.getpid()}

    def op_close(self) -> dict:
        if self._bridge is not None:
            self._bridge.close()
            self._bridge = None
        return {"closed": True}

    # ------------------------------------------------------------------
    # 分派
    # ------------------------------------------------------------------

    _OPS = {
        "load_csv": "op_load_csv",
        "sql": "op_sql",
        "python": "op_python",
        "tables": "op_tables",
        "table_details": "op_table_details",
        "ping": "op_ping",
        "close": "op_close",
    }

    def _require_bridge(self) -> CsvDatabaseBridge:
        if self._bridge is None:
            raise RuntimeError("no dataset loaded — call load_csv first")
        return self._bridge

    def handle(self, req: dict) -> dict:
        op = req.get("op")
        method = self._OPS.get(op)
        if method is None:
            raise ValueError(f"unknown op: {op!r} (known: {sorted(self._OPS)})")
        return getattr(self, method)(**(req.get("args") or {}))


def main() -> None:
    worker = Worker()
    # 就绪信号走 stderr，避免污染 stdout 协议通道
    print("WORKER_READY", file=sys.stderr, flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req_id = None
        try:
            req = json.loads(line)
            req_id = req.get("id")
            resp = {"id": req_id, "ok": True, "result": worker.handle(req)}
        except Exception as e:
            resp = {
                "id": req_id,
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(limit=6),
            }
        sys.stdout.write(json.dumps(_sanitize_unicode(resp), ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
