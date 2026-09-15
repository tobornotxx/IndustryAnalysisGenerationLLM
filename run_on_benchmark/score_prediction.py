"""Score an existing prediction without rerunning its generating agent."""

from __future__ import annotations

import argparse
import importlib
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

from .experiment_io import read_json, write_json_exclusive
from .deterministic_metrics import evaluate_deterministic


def load_ground_truth(benchmark_dir: Path, case_id: str) -> dict[str, Any]:
    path = benchmark_dir / "data" / "notebooks" / f"{case_id}.json"
    data = read_json(path)
    return {"insights": data.get("insights") or [], "summary": data.get("summary") or ""}


def score_prediction(
    *, prediction_path: Path, benchmark_dir: Path, judge_run: int,
    scorer_id: str = "local-deepseek-v41", scorer_module: ModuleType | Any | None = None,
) -> Path:
    if judge_run < 1:
        raise ValueError("judge_run must be >= 1")
    prediction = read_json(prediction_path)
    case_id = str(prediction.get("case_id") or "")
    if not case_id:
        raise ValueError("prediction is missing case_id")
    gt = load_ground_truth(benchmark_dir, case_id)
    scorer = scorer_module or importlib.import_module("run_on_benchmark.unified_scorer")

    modes = {
        "primary": prediction.get("pred_insights") or [],
        "raw": prediction.get("pred_insights_raw") or [],
        "bank": prediction.get("pred_insights_bank") or [],
    }
    semantic = {}
    for name, items in modes.items():
        if items:
            result = scorer.score_insight_matrix(items, gt["insights"])
            semantic[name] = {
                "recall": result["recall"], "precision": result["precision"],
                "f1": result["f1"], "n_pred": len(items), "matrix": result["matrix"],
            }
    pred_summary = prediction.get("pred_summary") or ""
    summary_score = (
        scorer.score_summary(pred_summary, gt["summary"])
        if pred_summary and gt["summary"] else None
    )
    output = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prediction": str(prediction_path.resolve()),
        "case_id": case_id,
        "judge_run": judge_run,
        "scorer_id": scorer_id,
        "scorer": scorer.get_scorer_config(),
        "semantic": semantic,
        "summary": summary_score,
        "deterministic": evaluate_deterministic(prediction, gt),
        "usage": scorer.get_usage_stats(),
    }
    output_path = prediction_path.parent / "scores" / scorer_id / f"judge_run_{judge_run}.json"
    write_json_exclusive(output_path, output)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--judge-run", type=int, required=True)
    parser.add_argument("--scorer-id", default="local-deepseek-v41")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = score_prediction(
        prediction_path=args.prediction, benchmark_dir=args.benchmark_dir,
        judge_run=args.judge_run, scorer_id=args.scorer_id,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
