import unittest

from run_on_benchmark.deterministic_metrics import (
    evaluate_deterministic, numeric_coverage, rouge_1, rouge_l, token_f1, two_way_token_overlap,
)


class DeterministicMetricsTests(unittest.TestCase):
    def test_token_f1_and_two_way_overlap(self):
        self.assertEqual(token_f1("Sales increased 20%", "sales increased 20%"), 1.0)
        result = two_way_token_overlap(["alpha beta", "gamma"], ["alpha beta"])
        self.assertEqual(result["recall"], 1.0)
        self.assertLess(result["precision"], 1.0)

    def test_rouge_l_respects_sequence(self):
        exact = rouge_l("a b c", "a b c")
        reordered = rouge_l("c b a", "a b c")
        self.assertEqual(exact["f1"], 1.0)
        self.assertLess(reordered["f1"], exact["f1"])
        self.assertEqual(rouge_1("c b a", "a b c")["f1"], 1.0)

    def test_numeric_coverage_handles_commas_percent_and_tolerance(self):
        result = numeric_coverage(["Revenue was 1,005 and margin 20.1%."], ["Revenue was 1,000 and margin 20%."])
        self.assertEqual(result["matched"], 2)
        self.assertEqual(result["coverage"], 1.0)

    def test_complete_bundle_contains_no_model_dependent_metric(self):
        result = evaluate_deterministic(
            {"pred_insights": ["Sales rose from 10 to 20."], "pred_summary": "Sales rose."},
            {"insights": ["Sales increased to 20."], "summary": "Sales rose strongly."},
        )
        self.assertEqual(set(result), {"token_overlap", "summary_rouge_1", "summary_rouge_l", "numeric_coverage", "output_profile"})
        self.assertEqual(result["output_profile"]["n_insights"], 1)


if __name__ == "__main__":
    unittest.main()
