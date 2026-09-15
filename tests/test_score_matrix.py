import tempfile
import unittest
from pathlib import Path

from run_on_benchmark.score_matrix import plan_score_tasks


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


if __name__ == "__main__":
    unittest.main()
