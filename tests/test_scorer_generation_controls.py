import unittest
from unittest.mock import patch

from run_on_benchmark import unified_scorer


class ScorerGenerationControlTests(unittest.TestCase):
    def test_thinking_mode_keeps_reasoning_budget(self):
        with (
            patch.object(unified_scorer, "_SCORER_THINKING", "enabled"),
            patch.object(unified_scorer, "_SCORER_MAX_TOKENS", 4096),
        ):
            controls = unified_scorer._generation_controls(50)
        self.assertEqual(controls["max_tokens"], 4096)
        self.assertEqual(controls["extra_body"], {"thinking": {"type": "enabled"}})

    def test_nonthinking_mode_keeps_short_rating_budget(self):
        with (
            patch.object(unified_scorer, "_SCORER_THINKING", "disabled"),
            patch.object(unified_scorer, "_SCORER_MAX_TOKENS", 4096),
        ):
            controls = unified_scorer._generation_controls(50)
        self.assertEqual(controls["max_tokens"], 50)
        self.assertEqual(controls["extra_body"], {"thinking": {"type": "disabled"}})


if __name__ == "__main__":
    unittest.main()
