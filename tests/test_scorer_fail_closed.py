from unittest import TestCase
from unittest.mock import Mock, patch

from run_on_benchmark import unified_scorer


class ScorerFailClosedTests(TestCase):
    def test_monte_carlo_raises_when_every_call_fails(self):
        client = Mock()
        client.chat.completions.create.side_effect = RuntimeError("transport down")

        with patch.object(unified_scorer, "_MC_SAMPLES", 2):
            with self.assertRaisesRegex(RuntimeError, "all Monte Carlo judge calls failed"):
                unified_scorer._score_pair_monte_carlo(
                    client, "deepseek-flash", "prediction", "reference"
                )

    def test_unexpected_logprobs_probe_error_is_not_downgraded(self):
        client = Mock()
        with (
            patch.object(unified_scorer, "_logprobs_supported", None),
            patch.object(
                unified_scorer,
                "_detect_logprobs",
                side_effect=RuntimeError("authentication failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "authentication failed"):
                unified_scorer._resolve_score_func(client, "deepseek-flash")
