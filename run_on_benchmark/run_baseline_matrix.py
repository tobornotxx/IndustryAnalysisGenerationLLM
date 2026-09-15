"""Run repeated immutable DataSTORM and AgentPoirot generations."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .generate_baseline import DATA_SPLITS, SUPPORTED_BENCHMARKS, SUPPORTED_SYSTEMS, run_directory


def plan_tasks(
    *, out_root: Path, experiment: str, systems: list[str], cases: list[str], agent_runs: int,
) -> list[dict]:
    if agent_runs < 1:
        raise ValueError("agent_runs must be >= 1")
    tasks = []
    for system in systems:
        if system not in SUPPORTED_SYSTEMS:
            raise ValueError(f"unsupported system: {system}")
        for case in cases:
            for run in range(1, agent_runs + 1):
                path = run_directory(out_root, experiment, system, case, run)
                manifest = path / "manifest.json"
                status = None
                if manifest.is_file():
                    try:
                        status = json.loads(manifest.read_text(encoding="utf-8")).get("status", "unknown")
                    except (OSError, ValueError):
                        status = "corrupt"
                elif path.exists():
                    status = "incomplete"
                tasks.append({"system": system, "case": case, "agent_run": run, "run_dir": path, "existing_status": status})
    return tasks


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", default=",".join(SUPPORTED_SYSTEMS))
    parser.add_argument("--cases", required=True)
    parser.add_argument("--agent-runs", type=int, default=2)
    parser.add_argument("--benchmark-kind", choices=SUPPORTED_BENCHMARKS, default="insightbench")
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--split", choices=DATA_SPLITS, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--out-root", type=Path, default=Path("results/experiments"))
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--max-layers", type=int, default=3)
    parser.add_argument("--questions", type=int, default=2)
    parser.add_argument("--max-insights", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    systems = [item for item in args.systems.split(",") if item]
    prefix = "insighteval-" if args.benchmark_kind == "insighteval" else "flag-"
    cases = [item if item.startswith(prefix) else f"{prefix}{item}" for item in args.cases.split(",") if item]
    tasks = plan_tasks(
        out_root=args.out_root, experiment=args.experiment, systems=systems,
        cases=cases, agent_runs=args.agent_runs,
    )
    if args.dry_run:
        print(json.dumps({
            "experiment_id": args.experiment,
            "config": {"model": args.model, "max_layers": args.max_layers, "questions": args.questions, "max_insights": args.max_insights},
            "tasks": [{**task, "run_dir": str(task["run_dir"])} for task in tasks],
        }, ensure_ascii=False, indent=2))
        return 0
    counts = {"success": 0, "failed": 0, "skipped": 0}
    for task in tasks:
        if task["existing_status"]:
            counts["skipped"] += 1
            continue
        result = subprocess.run([
            sys.executable, "-m", "run_on_benchmark.generate_baseline",
            "--system", task["system"], "--benchmark-kind", args.benchmark_kind,
            "--benchmark-dir", str(args.benchmark_dir), "--case", task["case"],
            "--split", args.split, "--experiment", args.experiment,
            "--agent-run", str(task["agent_run"]), "--out-root", str(args.out_root),
            "--model", args.model, "--max-layers", str(args.max_layers),
            "--questions", str(args.questions), "--max-insights", str(args.max_insights),
        ], check=False)
        counts["success" if result.returncode == 0 else "failed"] += 1
    print(json.dumps(counts, indent=2))
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
