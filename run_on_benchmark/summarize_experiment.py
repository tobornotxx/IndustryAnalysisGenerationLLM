"""Case-level experiment aggregation and paired uncertainty estimates."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .experiment_io import read_json, write_json_exclusive


def nested(value: dict, dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def sample_std(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) >= 2 else None


def bootstrap_ci(differences: list[float], samples: int = 10_000, seed: int = 20260915) -> list[float] | None:
    if not differences:
        return None
    rng = random.Random(seed)
    estimates = sorted(
        sum(rng.choice(differences) for _ in differences) / len(differences)
        for _ in range(samples)
    )
    return [estimates[int(0.025 * (samples - 1))], estimates[int(0.975 * (samples - 1))]]


def paired_permutation_p(differences: list[float], samples: int = 10_000, seed: int = 20260915) -> float | None:
    if not differences:
        return None
    observed = abs(sum(differences) / len(differences))
    if len(differences) <= 18:
        values = (
            abs(sum(sign * value for sign, value in zip(signs, differences)) / len(differences))
            for signs in itertools.product((-1, 1), repeat=len(differences))
        )
        extreme = total = 0
        for value in values:
            total += 1
            extreme += value >= observed - 1e-15
        return extreme / total
    rng = random.Random(seed)
    extreme = 0
    for _ in range(samples):
        value = abs(sum(rng.choice((-1, 1)) * item for item in differences) / len(differences))
        extreme += value >= observed - 1e-15
    return (extreme + 1) / (samples + 1)


def summarize_experiment(
    experiment_dir: Path,
    *,
    metric: str = "semantic.primary.f1",
    scorer_id: str = "local-deepseek-v41",
    baseline: str | None = None,
    challenger: str | None = None,
) -> dict:
    runs = []
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    status_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for manifest_path in sorted(experiment_dir.rglob("manifest.json")):
        manifest = read_json(manifest_path)
        system = str(manifest.get("system_id") or manifest_path.parents[2].name)
        case = str(manifest.get("case_id") or manifest_path.parents[1].name)
        status = str(manifest.get("status") or "unknown")
        status_counts[system][status] += 1
        score_paths = sorted((manifest_path.parent / "scores" / scorer_id).glob("judge_run_*.json"))
        judge_values = []
        for score_path in score_paths:
            value = nested(read_json(score_path), metric)
            if isinstance(value, (int, float)) and math.isfinite(value):
                judge_values.append(float(value))
        agent_value = mean(judge_values)
        if status == "success" and agent_value is not None:
            grouped[(system, case)].append(agent_value)
        runs.append({
            "system_id": system, "case_id": case, "status": status,
            "agent_run": manifest.get("agent_run"), "n_judge_runs": len(judge_values),
            "agent_mean": agent_value, "judge_std": sample_std(judge_values),
        })

    case_rows = []
    systems: dict[str, list[float]] = defaultdict(list)
    case_lookup: dict[str, dict[str, float]] = defaultdict(dict)
    for (system, case), agent_values in sorted(grouped.items()):
        case_mean = mean(agent_values)
        assert case_mean is not None
        systems[system].append(case_mean)
        case_lookup[system][case] = case_mean
        case_rows.append({
            "system_id": system, "case_id": case, "n_agent_runs": len(agent_values),
            "case_mean": case_mean, "agent_std": sample_std(agent_values),
        })

    system_rows = {
        system: {
            "n_cases": len(values), "mean": mean(values), "case_std": sample_std(values),
            "statuses": dict(status_counts[system]),
        }
        for system, values in sorted(systems.items())
    }
    for system in status_counts:
        system_rows.setdefault(system, {
            "n_cases": 0, "mean": None, "case_std": None, "statuses": dict(status_counts[system]),
        })

    comparison = None
    if baseline and challenger:
        common = sorted(set(case_lookup[baseline]) & set(case_lookup[challenger]))
        differences = [case_lookup[challenger][case] - case_lookup[baseline][case] for case in common]
        sd = sample_std(differences)
        comparison = {
            "baseline": baseline, "challenger": challenger, "n_paired_cases": len(common),
            "cases": common, "differences": differences, "mean_difference": mean(differences),
            "bootstrap_95_ci": bootstrap_ci(differences),
            "paired_permutation_p": paired_permutation_p(differences),
            "cohen_dz": (mean(differences) / sd) if differences and sd not in (None, 0) else None,
        }

    judge_stds = [row["judge_std"] for row in runs if row["judge_std"] is not None]
    agent_stds = [row["agent_std"] for row in case_rows if row["agent_std"] is not None]
    return {
        "schema_version": 1, "metric": metric, "scorer_id": scorer_id,
        "systems": system_rows, "case_results": case_rows, "runs": runs,
        "variance": {
            "mean_within_output_judge_std": mean(judge_stds),
            "mean_within_case_agent_std": mean(agent_stds),
        },
        "paired_comparison": comparison,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--metric", default="semantic.primary.f1")
    parser.add_argument("--scorer-id", default="local-deepseek-v41")
    parser.add_argument("--baseline")
    parser.add_argument("--challenger")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if bool(args.baseline) != bool(args.challenger):
        raise SystemExit("--baseline and --challenger must be provided together")
    summary = summarize_experiment(
        args.experiment_dir, metric=args.metric, scorer_id=args.scorer_id,
        baseline=args.baseline, challenger=args.challenger,
    )
    if args.output:
        write_json_exclusive(args.output, summary)
        print(args.output)
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
