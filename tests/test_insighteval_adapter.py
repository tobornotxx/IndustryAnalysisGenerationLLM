import unittest
import json
import tempfile
from pathlib import Path

from run_on_benchmark.insighteval_adapter import evaluate_prediction, load_instance, main


class InsightEvalAdapterTests(unittest.TestCase):
    def test_loads_pinned_official_instance_as_target_test(self):
        instance = load_instance(1)
        self.assertEqual(instance["case_id"], "insighteval-1")
        self.assertEqual(instance["split"], "target-test")
        self.assertTrue(instance["csv_path"].endswith("data_1.csv"))
        self.assertEqual(len(instance["reference_insights"]), 10)

    def test_calls_official_local_metric_without_api(self):
        instance = load_instance(1)
        result = evaluate_prediction(
            {"pred_insights": instance["reference_insights"]}, instance
        )
        self.assertEqual(result["metric"], "rouge1_f1_best_match")
        self.assertAlmostEqual(result["insight_recall"], 1.0)
        self.assertAlmostEqual(result["insight_precision"], 1.0)
        self.assertAlmostEqual(result["insight_f1"], 1.0)
        self.assertNotEqual(result["metric_source_commit"], "unknown")

    def test_cli_writes_score_once_without_api(self):
        instance = load_instance(1)
        with tempfile.TemporaryDirectory() as tmp:
            prediction = Path(tmp) / "prediction.json"
            output = Path(tmp) / "insighteval_score.json"
            prediction.write_text(
                json.dumps({"pred_insights": instance["reference_insights"]}), encoding="utf-8"
            )
            self.assertEqual(main(["--instance", "1", "--prediction", str(prediction), "--output", str(output)]), 0)
            self.assertAlmostEqual(json.loads(output.read_text(encoding="utf-8"))["insight_f1"], 1.0)
            with self.assertRaises(FileExistsError):
                main(["--instance", "1", "--prediction", str(prediction), "--output", str(output)])


if __name__ == "__main__":
    unittest.main()
