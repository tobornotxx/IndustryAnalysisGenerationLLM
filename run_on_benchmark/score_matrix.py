"""Schedule repeated Judge runs for every fixed prediction in an experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .experiment_io import write_json_exclusive
from .scorer_config import DEFAULT_SCORER_ID


def plan_score_tasks(experiment_dir: Path, judge_runs: int, scorer_id: str) -> list[dict]:
    if judge_runs < 1:
        raise ValueError("judge_runs must be >= 1")
    tasks = []
    for prediction in sorted(experiment_dir.rglob("prediction.json")):
        for judge_run in range(1, judge_runs + 1):
            output = prediction.parent / "scores" / scorer_id / f"judge_run_{judge_run}.json"
            tasks.append({
                "prediction": prediction,
                "judge_run": judge_run,
                "output": output,
                "existing": output.exists(),
            })
    return tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--judge-runs", type=int, default=3)
    parser.add_argument("--scorer-id", default=DEFAULT_SCORER_ID)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue after a failed judge process (default: stop after the first failure).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tasks = plan_score_tasks(args.experiment_dir, args.judge_runs, args.scorer_id)
    if args.dry_run:
        print(json.dumps([
            {**task, "prediction": str(task["prediction"]), "output": str(task["output"])}
            for task in tasks
        ], ensure_ascii=False, indent=2))
        return 0
    counts = {"success": 0, "failed": 0, "skipped": 0}
    for task_index, task in enumerate(tasks):
        if task["existing"]:
            counts["skipped"] += 1
            continue
        result = subprocess.run([
            sys.executable, "-m", "run_on_benchmark.score_prediction",
            "--prediction", str(task["prediction"]),
            "--benchmark-dir", str(args.benchmark_dir),
            "--judge-run", str(task["judge_run"]),
            "--scorer-id", args.scorer_id,
        ], check=False, capture_output=True, text=True)
        if result.returncode == 0:
            counts["success"] += 1
            print(result.stdout.strip())
        else:
            counts["failed"] += 1
            failure = task["output"].with_suffix(".failure.json")
            if not failure.exists():
                write_json_exclusive(failure, {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "prediction": str(task["prediction"].resolve()),
                    "judge_run": task["judge_run"],
                    "returncode": result.returncode,
                    "stdout": result.stdout[-4000:],
                    "stderr": result.stderr[-4000:],
                })
            print(result.stderr.strip(), file=sys.stderr)
            if not args.keep_going:
                counts["not_run"] = sum(
                    not remaining["existing"] for remaining in tasks[task_index + 1:]
                )
                break
    print(json.dumps(counts, indent=2))
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
