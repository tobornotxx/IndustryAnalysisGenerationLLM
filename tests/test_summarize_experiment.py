import json
import tempfile
import unittest
from pathlib import Path

from run_on_benchmark.summarize_experiment import (
    bootstrap_ci, paired_permutation_p, summarize_experiment,
)


def write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class SummaryTests(unittest.TestCase):
    def test_statistics_are_case_level_after_judge_and_agent_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for system, shift in (("base", 0.0), ("new", 0.1)):
                for case_index in (1, 2):
                    for agent_run in (1, 2):
                        run = root / system / f"flag-{case_index}" / f"agent_run_{agent_run}"
                        write_json(run / "manifest.json", {
                            "system_id": system, "case_id": f"flag-{case_index}",
                            "agent_run": agent_run, "status": "success",
                        })
                        for judge_run, noise in ((1, -0.01), (2, 0.01)):
                            write_json(run / "scores" / "judge" / f"judge_run_{judge_run}.json", {
                                "semantic": {"primary": {"f1": 0.5 + 0.05 * case_index + shift + noise}}
                            })
                        write_json(run / "usage.json", {
                            "cost_usd": 0.2 if system == "base" else 0.3,
                            "prompt_tokens": 1000, "completion_tokens": 200,
                        })
            result = summarize_experiment(
                root, scorer_id="judge", baseline="base", challenger="new"
            )
        self.assertEqual(result["paired_comparison"]["n_paired_cases"], 2)
        self.assertAlmostEqual(result["paired_comparison"]["mean_difference"], 0.1)
        self.assertEqual(result["systems"]["base"]["n_cases"], 2)
        self.assertGreater(result["variance"]["mean_within_output_judge_std"], 0)
        self.assertAlmostEqual(result["paired_comparison"]["mean_cost_ratio"], 1.5)
        self.assertAlmostEqual(result["systems"]["base"]["mean_cost_usd_per_case"], 0.2)
        self.assertEqual(result["systems"]["base"]["mean_tokens_per_case"], 1200)
        self.assertIsNotNone(result["systems"]["base"]["quality_per_usd"])

    def test_failed_runs_remain_in_status_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_json(root / "system" / "flag-1" / "agent_run_1" / "manifest.json", {
                "system_id": "system", "case_id": "flag-1", "status": "failed",
            })
            result = summarize_experiment(root)
        self.assertEqual(result["systems"]["system"]["statuses"]["failed"], 1)
        self.assertEqual(result["systems"]["system"]["n_cases"], 0)

    def test_resampling_functions_are_deterministic(self):
        values = [0.1, -0.02, 0.05]
        self.assertEqual(bootstrap_ci(values), bootstrap_ci(values))
        self.assertEqual(paired_permutation_p(values), paired_permutation_p(values))


if __name__ == "__main__":
    unittest.main()
