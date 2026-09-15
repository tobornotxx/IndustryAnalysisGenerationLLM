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

    def test_environment_has_priority_and_alias_is_normalized(self):
        cfg = load_scorer_config(
            env={"SCORER_MODEL": "deepseek-v4-pro", "SCORER_API_KEY": "judge-key"}
        )
        self.assertEqual(cfg["model"], "deepseek-flash")
        self.assertEqual(cfg["api_key"], "judge-key")


if __name__ == "__main__":
    unittest.main()
