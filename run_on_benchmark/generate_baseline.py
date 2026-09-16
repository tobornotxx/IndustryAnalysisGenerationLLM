"""Generate immutable predictions for locally executed Python baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .experiment_io import read_json, write_json_atomic, write_json_exclusive

SUPPORTED_SYSTEMS = ("legacy-custom-python", "agentpoirot-upstream-local")
SUPPORTED_BENCHMARKS = ("insightbench", "insighteval")
DATA_SPLITS = ("dev-contaminated", "source-train", "source-valid", "source-test", "target-test")


def git_state(path: Path) -> dict:
    try:
        safe = f"safe.directory={path.resolve()}"
        commit = subprocess.run(["git", "-c", safe, "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-c", safe, "status", "--porcelain"], cwd=path, check=True, capture_output=True, text=True).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": "unknown", "dirty": None}


def sha256_files(paths: list[Path]) -> str | None:
    existing = sorted(path for path in paths if path.is_file())
    if not existing:
        return None
    digest = hashlib.sha256()
    for path in existing:
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def run_directory(out_root: Path, experiment: str, system: str, case: str, agent_run: int) -> Path:
    return out_root / experiment / system / case / f"agent_run_{agent_run}"


def load_benchmark_case(benchmark_kind: str, benchmark_dir: Path, case_id: str) -> dict[str, Any]:
    if benchmark_kind == "insighteval":
        from .insighteval_adapter import load_instance
        instance_id = int(case_id.removeprefix("insighteval-"))
        data_dir = benchmark_dir / "data" if (benchmark_dir / "data").is_dir() else benchmark_dir
        return load_instance(instance_id, data_dir)
    meta_path = benchmark_dir / "data" / "notebooks" / f"{case_id}.json"
    meta = read_json(meta_path)
    return {
        "benchmark_id": "insightbench-overhaul",
        "case_id": case_id,
        "goal": (meta.get("metadata") or {}).get("goal") or "Find interesting trends in this dataset",
        "dataset_description": (meta.get("metadata") or {}).get("dataset_description", ""),
        "csv_path": str((benchmark_dir / meta["dataset_csv_path"]).resolve()),
        "user_csv_path": (
            str((benchmark_dir / meta["user_dataset_csv_path"]).resolve())
            if meta.get("user_dataset_csv_path") else None
        ),
    }


def generate(
    *, system: str, benchmark_dir: Path, case_id: str, run_dir: Path,
    model: str, max_layers: int, questions: int, max_questions: int = 6,
    max_insights: int = 10,
    benchmark_kind: str = "insightbench",
    runner: Callable[..., dict] | None = None,
) -> dict:
    case = load_benchmark_case(benchmark_kind, benchmark_dir, case_id)
    goal = case["goal"]
    csv_path = Path(case["csv_path"])
    user_csv = Path(case["user_csv_path"]) if case.get("user_csv_path") else None
    if runner:
        return runner(system=system, csv_path=csv_path, user_csv=user_csv, goal=goal, run_dir=run_dir)
    if system == "agentpoirot-upstream-local":
        from .agentpoirot_adapter import run_agentpoirot
        output = run_agentpoirot(
            csv_path=csv_path, goal=goal, output_dir=run_dir / "system_artifacts",
            n_insights=max_insights, model=model,
        )
        output["case_id"] = case_id
        return output
    if system == "legacy-custom-python":
        from .datastorm_adapter.adapter import DataStormAdapter
        artifacts = run_dir / "system_artifacts"
        artifacts.mkdir()
        adapter = DataStormAdapter(
            model_name=model, max_layers=max_layers, questions_per_layer=questions,
            max_questions=max_questions, max_insights=max_insights,
            savedir=str(artifacts), summary_samples=3,
        )
        insights, summary = adapter.get_insights(
            dataset_csv_path=str(csv_path), user_dataset_csv_path=str(user_csv) if user_csv else None,
            goal=goal, dataset_description=case.get("dataset_description", ""),
            return_summary=True,
        )
        return {
            "schema_version": 1, "system_id": system, "case_id": case_id,
            "generation_model": model, "goal": goal,
            "pred_insights": insights, "pred_summary": summary,
        }
    raise ValueError(f"unsupported system: {system}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=SUPPORTED_SYSTEMS, required=True)
    parser.add_argument("--benchmark-kind", choices=SUPPORTED_BENCHMARKS, default="insightbench")
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--experiment", default="v41_baseline")
    parser.add_argument("--split", choices=DATA_SPLITS)
    parser.add_argument("--agent-run", type=int, default=1)
    parser.add_argument("--out-root", type=Path, default=Path("results/experiments"))
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--max-layers", type=int, default=3)
    parser.add_argument("--questions", type=int, default=2)
    parser.add_argument("--max-questions", type=int, default=6)
    parser.add_argument("--max-insights", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.benchmark_kind == "insighteval":
        case_id = args.case if args.case.startswith("insighteval-") else f"insighteval-{args.case}"
        split = args.split or "target-test"
        if split != "target-test":
            raise SystemExit("InsightEval is reserved for target-test and must not be used for tuning")
    else:
        case_id = args.case if args.case.startswith("flag-") else f"flag-{args.case}"
        split = args.split or "dev-contaminated"
    try:
        case = load_benchmark_case(args.benchmark_kind, args.benchmark_dir, case_id)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    run_dir = run_directory(args.out_root, args.experiment, args.system, case_id, args.agent_run)
    repository_root = Path(__file__).resolve().parents[1]
    if args.system == "agentpoirot-upstream-local":
        system_root = Path(__file__).with_name("agent-poirot")
        prompt_files = list((system_root / "agentpoirot" / "prompts").glob("*.txt"))
    else:
        system_root = repository_root.parent / "MyDataStorm"
        prompt_files = [system_root / "datastorm" / "prompts" / "templates.py"]
    plan = {
        "schema_version": 1, "run_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(), "status": "planned",
        "experiment_id": args.experiment, "system_id": args.system, "case_id": case_id,
        "benchmark_id": case["benchmark_id"], "split": split,
        "agent_run": args.agent_run, "generation_model": args.model, "scorer_model": None,
        "benchmark": git_state(args.benchmark_dir),
        "repository": git_state(repository_root), "system_source": git_state(system_root),
        "prompt_hash": sha256_files(prompt_files),
        "config": {"max_layers": args.max_layers, "questions_per_layer": args.questions, "max_questions": args.max_questions, "max_insights": args.max_insights},
    }
    if args.dry_run:
        print(json.dumps({"run_directory": str(run_dir.resolve()), "manifest": plan}, ensure_ascii=False, indent=2))
        return 0
    if run_dir.exists():
        raise SystemExit(f"refusing to overwrite existing run: {run_dir}")
    run_dir.mkdir(parents=True)
    write_json_atomic(run_dir / "manifest.json", {**plan, "status": "running"})
    started = time.perf_counter()
    try:
        prediction = generate(
            system=args.system, benchmark_dir=args.benchmark_dir, case_id=case_id,
            run_dir=run_dir, model=args.model, max_layers=args.max_layers, questions=args.questions,
            max_questions=args.max_questions, max_insights=args.max_insights,
            benchmark_kind=args.benchmark_kind,
        )
        prediction.setdefault("benchmark_id", case["benchmark_id"])
        prediction.setdefault("case_id", case_id)
        prediction.setdefault("split", split)
        write_json_exclusive(run_dir / "prediction.json", prediction)
        write_json_atomic(run_dir / "manifest.json", {
            **plan, "status": "success", "completed_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_sec": round(time.perf_counter() - started, 3),
        })
        print(run_dir)
        return 0
    except Exception as error:
        write_json_atomic(run_dir / "manifest.json", {
            **plan, "status": "failed", "completed_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()[-8000:]},
        })
        raise


if __name__ == "__main__":
    raise SystemExit(main())
