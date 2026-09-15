import tempfile
import unittest
from pathlib import Path

from run_on_benchmark.scorer_config import load_scorer_config


class ScorerConfigTests(unittest.TestCase):
    def test_business_model_does_not_control_judge(self):
        with tempfile.TemporaryDirectory() as tmp:
            legacy = Path(tmp) / "legacy.json"
            legacy.write_text(
                '{"default":{"model_name":"business-model","api_key":"legacy-key"}}',
                encoding="utf-8",
            )
            cfg = load_scorer_config(env={}, legacy_path=legacy)
        self.assertEqual(cfg["model"], "deepseek-flash")
        self.assertEqual(cfg["api_key"], "legacy-key")
        self.assertEqual(cfg["thinking"], "disabled")
        self.assertEqual(cfg["max_tokens"], 50)

    def test_environment_has_priority_and_alias_is_normalized(self):
        cfg = load_scorer_config(
            env={"SCORER_MODEL": "deepseek-v4-pro", "SCORER_API_KEY": "judge-key"}
        )
        self.assertEqual(cfg["model"], "deepseek-flash")
        self.assertEqual(cfg["api_key"], "judge-key")

    def test_explicit_thinking_mode_is_recorded(self):
        cfg = load_scorer_config(env={"SCORER_THINKING": "enabled", "SCORER_MAX_TOKENS": "20"})
        self.assertEqual(cfg["thinking"], "enabled")
        self.assertEqual(cfg["max_tokens"], 20)

    def test_invalid_thinking_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            load_scorer_config(env={"SCORER_THINKING": "sometimes"})


if __name__ == "__main__":
    unittest.main()
