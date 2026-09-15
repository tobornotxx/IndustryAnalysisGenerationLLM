import unittest

from run_on_benchmark.insighteval_adapter import evaluate_prediction, load_instance


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


if __name__ == "__main__":
    unittest.main()
