"""Thin adapter around the pinned, official InsightEval benchmark.

The benchmark data and metric implementation remain in the upstream checkout.
This module only normalizes them to the experiment artifact schema used here.
No model or evaluator API is invoked.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any


OFFICIAL_ROOT = Path(__file__).with_name("InsightEval-official")


def _resolve_table_path(data_dir: Path, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    candidate = data_dir / raw_path.lstrip("./")
    if candidate.is_file():
        return candidate.resolve()
    alternate = candidate.with_name(
        candidate.name.replace("data-", "data_", 1).replace("-sysuser", "_sysuser")
    )
    if alternate.is_file():
        return alternate.resolve()
    raise FileNotFoundError(f"InsightEval table does not exist: {candidate}")


def load_instance(instance_id: int, data_dir: Path | None = None) -> dict[str, Any]:
    """Load an official instance without importing the agent or pandas runtime."""
    if not 1 <= instance_id <= 100:
        raise ValueError("instance_id must be between 1 and 100")
    root = (data_dir or OFFICIAL_ROOT / "data").resolve()
    annotation_path = root / "jsons" / f"data_{instance_id}.json"
    with annotation_path.open(encoding="utf-8") as handle:
        record = json.load(handle)
    metadata = record["metadata"]
    return {
        "benchmark_id": "insighteval-official",
        "case_id": f"insighteval-{instance_id}",
        "instance_id": instance_id,
        "split": "target-test",
        "goal": record["goal"],
        "dataset_description": metadata.get("table_description", ""),
        "category": metadata.get("category"),
        "difficulty": metadata.get("difficulty"),
        "csv_path": str(_resolve_table_path(root, metadata["table_path"])),
        "user_csv_path": (
            str(path) if (path := _resolve_table_path(root, metadata.get("user_table_path"))) else None
        ),
        "reference_insights": record["insights"],
        "reference_summary": record["summary"],
        "annotation_path": str(annotation_path.resolve()),
    }


def _official_metrics_module(root: Path = OFFICIAL_ROOT) -> ModuleType:
    metrics_path = root / "src" / "insighteval" / "metrics.py"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"official InsightEval metrics not found: {metrics_path}")
    spec = importlib.util.spec_from_file_location("_official_insighteval_metrics", metrics_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official InsightEval metrics: {metrics_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def official_commit(root: Path = OFFICIAL_ROOT) -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def evaluate_prediction(prediction: dict[str, Any], instance: dict[str, Any]) -> dict[str, Any]:
    """Run the official dependency-free InsightEval metric on fixed output."""
    predicted = prediction.get("pred_insights", prediction.get("insights", []))
    if not isinstance(predicted, list) or not all(isinstance(item, str) for item in predicted):
        raise ValueError("prediction must contain a string list in pred_insights or insights")
    metrics = _official_metrics_module().evaluate_predictions(
        predicted, instance["reference_insights"]
    )
    return {
        "schema_version": 1,
        "benchmark_id": "insighteval-official",
        "case_id": instance["case_id"],
        "split": "target-test",
        "metric_source": "zhenghaozhu23/InsightEval:src/insighteval/metrics.py",
        "metric_source_commit": official_commit(),
        "prediction_count": len(predicted),
        **metrics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=int, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--prediction", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    instance = load_instance(args.instance, args.data_dir)
    result: dict[str, Any] = instance
    if args.prediction:
        with args.prediction.open(encoding="utf-8") as handle:
            prediction = json.load(handle)
        result = evaluate_prediction(prediction, instance)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
