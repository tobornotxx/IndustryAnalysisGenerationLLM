"""Re-score fixed historical PI outputs to audit judge-scale compatibility."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import unified_scorer as scorer


def _metrics(matrix: np.ndarray) -> dict[str, float]:
    recall = float(matrix.max(axis=1).mean())
    precision = float(matrix.max(axis=0).mean())
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    return {"recall": recall, "precision": precision, "f1": f1}


def _case_record(path: Path) -> dict:
    historical = json.loads(path.read_text(encoding="utf-8"))
    old_matrix = np.asarray(historical["score_matrix"], dtype=float)
    scorer.reset_usage_stats()
    current = scorer.score_insight_matrix(
        historical["pred_insights"], historical["gt_insights"]
    )
    new_matrix = np.asarray(current["matrix"], dtype=float)
    old_metrics = _metrics(old_matrix)
    new_metrics = _metrics(new_matrix)
    old_flat = old_matrix.ravel()
    new_flat = new_matrix.ravel()
    correlation = float(np.corrcoef(old_flat, new_flat)[0, 1])
    return {
        "case_id": historical["flag"],
        "source": str(path.resolve()),
        "n_gt": len(historical["gt_insights"]),
        "n_pred": len(historical["pred_insights"]),
        "n_pairs": int(old_matrix.size),
        "historical_scorer": historical.get("scorer"),
        "current_scorer": scorer.get_scorer_config(),
        "historical": old_metrics,
        "current": new_metrics,
        "delta": {key: new_metrics[key] - old_metrics[key] for key in old_metrics},
        "pairwise": {
            "pearson_r": correlation,
            "mean_absolute_delta": float(np.abs(new_flat - old_flat).mean()),
            "median_absolute_delta": float(np.median(np.abs(new_flat - old_flat))),
            "mean_signed_delta": float((new_flat - old_flat).mean()),
            "zero_cells_current": int((new_matrix == 0).sum()),
        },
        "usage": scorer.get_usage_stats(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidates = sorted(args.input_dir.glob("flag-*_result.json"))
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    selected = candidates[: args.sample_size]
    if not selected:
        raise SystemExit("no historical result files found")

    records = [_case_record(path) for path in selected]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sampling": {
            "population": len(candidates),
            "sample_size": len(selected),
            "seed": args.seed,
            "selected": [path.stem.removesuffix("_result") for path in selected],
        },
        "cases": records,
        "aggregate": {
            "historical_f1_mean": float(np.mean([r["historical"]["f1"] for r in records])),
            "current_f1_mean": float(np.mean([r["current"]["f1"] for r in records])),
            "f1_delta_mean": float(np.mean([r["delta"]["f1"] for r in records])),
            "pairwise_pearson_mean": float(np.mean([r["pairwise"]["pearson_r"] for r in records])),
            "pairwise_mae_mean": float(np.mean([r["pairwise"]["mean_absolute_delta"] for r in records])),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
