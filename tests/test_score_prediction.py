import json
import tempfile
import unittest
from pathlib import Path

from run_on_benchmark.score_prediction import score_prediction


class FakeScorer:
    insight_calls = 0

    @staticmethod
    def score_insight_matrix(pred, gt):
        FakeScorer.insight_calls += 1
        return {"recall": 0.5, "precision": 0.4, "f1": 4 / 9, "matrix": [[0.5]]}

    @staticmethod
    def score_summary(pred, gt):
        return 0.6

    @staticmethod
    def get_scorer_config():
        return {"model": "fake-judge", "prompt_order": "pred_first"}

    @staticmethod
    def get_usage_stats():
        return {"calls": 2}


class OfflineScoreTests(unittest.TestCase):
    def setUp(self):
        FakeScorer.insight_calls = 0
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        case_dir = self.root / "benchmark" / "data" / "notebooks"
        case_dir.mkdir(parents=True)
        (case_dir / "flag-11.json").write_text(
            json.dumps({"insights": ["ground truth"], "summary": "summary"}), encoding="utf-8"
        )
        self.run_dir = self.root / "run"
        self.run_dir.mkdir()
        self.prediction = self.run_dir / "prediction.json"
        self.prediction.write_text(json.dumps({
            "case_id": "flag-11", "pred_insights": ["prediction"],
            "pred_insights_raw": ["prediction"], "pred_insights_bank": [],
            "pred_summary": "generated summary",
        }), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_scores_fixed_prediction_without_agent_call(self):
        path = score_prediction(
            prediction_path=self.prediction, benchmark_dir=self.root / "benchmark",
            judge_run=1, scorer_module=FakeScorer,
        )
        score = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(score["scorer"]["model"], "fake-judge")
        self.assertEqual(score["semantic"]["primary"]["recall"], 0.5)
        self.assertEqual(score["judge_run"], 1)
        self.assertEqual(len(score["prediction_sha256"]), 64)
        self.assertEqual(FakeScorer.insight_calls, 1)
        self.assertEqual(score["semantic"]["raw"]["reused_from"], "primary")

    def test_score_artifact_cannot_be_overwritten(self):
        kwargs = dict(
            prediction_path=self.prediction, benchmark_dir=self.root / "benchmark",
            judge_run=1, scorer_module=FakeScorer,
        )
        score_prediction(**kwargs)
        with self.assertRaises(FileExistsError):
            score_prediction(**kwargs)


if __name__ == "__main__":
    unittest.main()
