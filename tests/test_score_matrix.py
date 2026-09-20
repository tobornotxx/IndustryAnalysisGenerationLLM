import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from run_on_benchmark.score_matrix import main, plan_score_tasks


class ScoreMatrixTests(unittest.TestCase):
    def test_discovers_predictions_and_resumes_existing_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "system" / "flag-11" / "agent_run_1"
            run.mkdir(parents=True)
            (run / "prediction.json").write_text("{}", encoding="utf-8")
            existing = run / "scores" / "judge" / "judge_run_1.json"
            existing.parent.mkdir(parents=True)
            existing.write_text("{}", encoding="utf-8")
            tasks = plan_score_tasks(root, 3, "judge")
        self.assertEqual(len(tasks), 3)
        self.assertEqual([task["existing"] for task in tasks], [True, False, False])

    def test_stops_after_first_failed_judge_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for case in ("case-a", "case-b"):
                prediction = root / case / "prediction.json"
                prediction.parent.mkdir(parents=True)
                prediction.write_text("{}", encoding="utf-8")
            failed = subprocess.CompletedProcess([], 1, stdout="", stderr="auth failed")
            with patch("run_on_benchmark.score_matrix.subprocess.run", return_value=failed) as run:
                exit_code = main([
                    "--experiment-dir", str(root),
                    "--benchmark-dir", str(root),
                    "--scorer-id", "judge-v2",
                ])
            self.assertEqual(exit_code, 1)
            self.assertEqual(run.call_count, 1)
            self.assertIn("--modes", run.call_args.args[0])
            self.assertEqual(len(list(root.rglob("*.failure.json"))), 1)


if __name__ == "__main__":
    unittest.main()
