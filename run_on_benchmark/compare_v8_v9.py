"""v8 / v9 同条件对比报告（flag-4, max_layers=5）。

用法: python compare_v8_v9.py
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
RUNS = {
    "v8  (L5, answer_first)": RESULTS / "test_run_v8" / "flag-4",
    "v9  (L3, gt_first)": RESULTS / "test_run_v9_flag4" / "flag-4",
    "v9  (L5, gt_first)": RESULTS / "test_run_v9_L5_flag4" / "flag-4",
}
LOGS = {
    "v8  (L5, answer_first)": RESULTS / "test_run_v8" / "flag-4" / "run.log",
    "v9  (L3, gt_first)": RESULTS / "test_run_v9_flag4" / "run_flag-4.log",
    "v9  (L5, gt_first)": RESULTS / "test_run_v9_L5_flag4" / "run_flag-4.log",
}


def load(p: Path) -> dict | None:
    f = p / "result.json"
    if not f.is_file():
        return None
    try:
        return json.loads(f.read_text())
    except Exception:
        return None


def count_calls(log: Path) -> int:
    if not log.is_file():
        return 0
    try:
        out = subprocess.run(
            ["grep", "-c", "httpx: HTTP Request", str(log)],
            capture_output=True, text=True,
        )
        return int((out.stdout or "0").strip() or 0)
    except Exception:
        return 0


def layers(log: Path) -> str:
    if not log.is_file():
        return "?"
    txt = log.read_text(errors="replace")
    ms = re.findall(r"Exploration Layer (\d+)/(\d+)", txt)
    return f"{ms[-1][0]}/{ms[-1][1]}" if ms else "?"


def main() -> None:
    rows = []
    for label, d in RUNS.items():
        r = load(d)
        if r is None:
            rows.append((label, None))
            continue
        rows.append((label, {
            "recall": r.get("insights_recall", r.get("score_insights")),
            "precision": r.get("insights_precision"),
            "f1": r.get("insights_f1"),
            "summary": r.get("score_summary"),
            "n_pred": r.get("n_pred_insights"),
            "n_gt": r.get("n_gt_insights"),
            "usage": r.get("scorer_usage") or {},
            "calls": count_calls(LOGS[label]),
            "layers": layers(LOGS[label]),
        }))

    w = 24
    print("=" * 96)
    print("flag-4 对比 (GT=8)".center(96))
    print("=" * 96)
    hdr = f"{'run':{w}s} {'layers':>7s} {'recall':>8s} {'prec':>8s} {'F1':>8s} {'summary':>8s} {'n_pred':>7s} {'calls':>7s}"
    print(hdr)
    print("-" * 96)
    for label, m in rows:
        if m is None:
            print(f"{label:{w}s}   (未完成/无结果)")
            continue
        def f(x):
            return f"{x:.4f}" if isinstance(x, (int, float)) and x is not None else "  n/a "
        print(f"{label:{w}s} {m['layers']:>7s} {f(m['recall']):>8s} {f(m['precision']):>8s} "
              f"{f(m['f1']):>8s} {f(m['summary']):>8s} {str(m['n_pred']):>7s} {m['calls']:>7d}")

    print()
    print("=" * 96)
    print("scorer token / 缓存".center(96))
    print("=" * 96)
    print(f"{'run':{w}s} {'order':>13s} {'calls':>6s} {'prompt':>10s} {'compl':>8s} {'cache_hit':>10s} {'hit%':>7s}")
    print("-" * 96)
    for label, m in rows:
        if m is None:
            continue
        u = m["usage"]
        if not u:
            print(f"{label:{w}s} {'(未记录)':>13s}")
            continue
        print(f"{label:{w}s} {u.get('prompt_order','?'):>13s} {u.get('calls',0):>6d} "
              f"{u.get('prompt_tokens',0):>10,d} {u.get('completion_tokens',0):>8,d} "
              f"{u.get('cache_hit_tokens',0):>10,d} {u.get('cache_hit_rate',0)*100:>6.1f}%")

    # 同条件对比（都是 L5）
    v8 = dict(rows)[list(RUNS)[0]]
    v9 = dict(rows)[list(RUNS)[2]]
    if v8 and v9:
        print()
        print("=" * 96)
        print("同条件 (均 L5) v8 → v9 变化".center(96))
        print("=" * 96)
        for k, name in [("recall", "recall"), ("summary", "summary"),
                        ("n_pred", "pred 条数"), ("calls", "LLM 调用")]:
            a, b = v8.get(k), v9.get(k)
            if a is None or b is None:
                continue
            if isinstance(a, float):
                print(f"  {name:12s} {a:.4f} → {b:.4f}   ({b-a:+.4f})")
            else:
                print(f"  {name:12s} {a} → {b}   ({b-a:+d})")
        print()
        print("  ⚠️ 单 case 结论有限：树形探索有运行间方差，v8 自身重跑也会波动。")
        print("     判断改动是否伤害分数需多 flag 或多次重复。")


if __name__ == "__main__":
    main()
