"""Generate a local prediction with pinned upstream AgentPoirot code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agentpoirot_adapter import run_agentpoirot, upstream_commit, write_prediction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--goal", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-insights", type=int, default=12)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite existing output directory: {args.out}")
    if args.dry_run:
        print(json.dumps({
            "system_id": "agentpoirot-upstream-local", "system_commit": upstream_commit(),
            "model": args.model, "csv": str(args.csv.resolve()), "goal": args.goal,
            "n_insights": args.n_insights,
        }, ensure_ascii=False, indent=2))
        return 0
    prediction = run_agentpoirot(
        csv_path=args.csv, goal=args.goal, output_dir=args.out,
        n_insights=args.n_insights, model=args.model,
    )
    write_prediction(args.out / "prediction.json", prediction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
