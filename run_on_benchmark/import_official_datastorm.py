"""Import author-published DataSTORM/AgentPoirot predictions without generation calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .experiment_io import read_json, write_json_exclusive
from .system_registry import system_record

OFFICIAL_COMMIT = "29ba2031290e40a388ce548582d4e43bbac05d72"
PUBLISHED_SYSTEMS = {
    "datastorm-official-published": "scores",
    "agentpoirot-official-published": "baseline_scores",
}


def _git_head(repo: Path) -> str:
    result = subprocess.run(
        ["git", "-c", f"safe.directory={repo.resolve()}", "rev-parse", "HEAD"],
        cwd=repo, check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _published_scores(raw: dict[str, Any]) -> dict[str, Any]:
    names = (
        "score_insights", "score_insights_Qwen3-30B-A3B-Instruct-2507",
        "score_llm_predicted_summary", "score_summary",
        "score_llm_predicted_summary_Qwen3-30B-A3B-Instruct-2507",
        "score_predicted_summary_Qwen3-30B-A3B-Instruct-2507",
    )
    return {name: raw[name] for name in names if name in raw}


def import_prediction(
    *, official_repo: Path, benchmark_dir: Path, out_root: Path,
    experiment_id: str, system_id: str, case_number: int,
) -> Path:
    if system_id not in PUBLISHED_SYSTEMS:
        raise ValueError(f"unsupported published system: {system_id}")
    if case_number < 1:
        raise ValueError("case_number must be >= 1")
    head = _git_head(official_repo)
    if head != OFFICIAL_COMMIT:
        raise ValueError(f"official DataSTORM source must be pinned to {OFFICIAL_COMMIT}; got {head}")

    source = (
        official_repo / "results" / "insight_bench" / PUBLISHED_SYSTEMS[system_id]
        / f"pred_gt_{case_number}.json"
    )
    raw = read_json(source)
    if int(raw.get("dataset_id", -1)) != case_number:
        raise ValueError(f"dataset id mismatch in {source}")
    case_id = f"flag-{case_number}"
    ground_truth = benchmark_dir / "data" / "notebooks" / f"{case_id}.json"
    if not ground_truth.is_file():
        raise FileNotFoundError(f"local ground truth not found: {ground_truth}")

    record = system_record(system_id)
    run_dir = out_root / experiment_id / system_id / case_id / "agent_run_1"
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing import: {run_dir}")
    now = datetime.now(timezone.utc).isoformat()
    summary = raw.get("llm_predicted_summary") or raw.get("predicted_summary") or ""
    prediction = {
        "schema_version": 1,
        "system_id": system_id,
        "case_id": case_id,
        "benchmark_id": "insightbench-overhaul",
        "split": "published-reference",
        "goal": (raw.get("metadata") or {}).get("goal", ""),
        "generation_model": record.get("generation_model"),
        "pred_insights": raw.get("predicted_insights") or [],
        "pred_summary": summary,
        "artifact_origin": "author-published",
    }
    manifest = {
        "schema_version": 1,
        "run_id": str(uuid.uuid4()),
        "created_at": now,
        "completed_at": now,
        "status": "success",
        "experiment_id": experiment_id,
        "system_id": system_id,
        "case_id": case_id,
        "benchmark_id": "insightbench-overhaul",
        "split": "published-reference",
        "agent_run": 1,
        "generation_model": record.get("generation_model"),
        "generation_mode": "published-artifact",
        "source": {
            "repository": record.get("repository"),
            "commit": head,
            "path": source.relative_to(official_repo).as_posix(),
            "sha256": _sha256(source),
        },
        "claim_boundary": record.get("claim_boundary"),
        "n_pred": len(prediction["pred_insights"]),
    }
    write_json_exclusive(run_dir / "prediction.json", prediction)
    write_json_exclusive(run_dir / "published_scores.json", {
        "schema_version": 1,
        "source_sha256": manifest["source"]["sha256"],
        "scores": _published_scores(raw),
    })
    write_json_exclusive(run_dir / "manifest.json", manifest)
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=Path("results/experiments"))
    parser.add_argument("--experiment", default="official_published_v1")
    parser.add_argument("--system", choices=tuple(PUBLISHED_SYSTEMS), required=True)
    parser.add_argument("--cases", required=True, help="comma-separated InsightBench numbers")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    case_numbers = [int(value) for value in args.cases.split(",") if value.strip()]
    for case_number in case_numbers:
        path = import_prediction(
            official_repo=args.official_repo, benchmark_dir=args.benchmark_dir,
            out_root=args.out_root, experiment_id=args.experiment,
            system_id=args.system, case_number=case_number,
        )
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
