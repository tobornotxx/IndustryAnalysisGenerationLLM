import json
import tempfile
import unittest
from pathlib import Path

from run_on_benchmark.generate_baseline import build_parser, generate, run_directory


class BaselineGenerationTests(unittest.TestCase):
    def test_common_run_path(self):
        path = run_directory(Path("results"), "exp", "agentpoirot-official", "flag-11", 2)
        self.assertEqual(path.as_posix(), "results/exp/agentpoirot-official/flag-11/agent_run_2")

    def test_runner_receives_resolved_common_inputs_without_api(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_dir = root / "data" / "notebooks"
            case_dir.mkdir(parents=True)
            (root / "data.csv").write_text("a\n1\n", encoding="utf-8")
            (case_dir / "flag-11.json").write_text(json.dumps({
                "dataset_csv_path": "data.csv", "metadata": {"goal": "test goal"},
            }), encoding="utf-8")
            calls = []
            def fake_runner(**kwargs):
                calls.append(kwargs)
                return {"pred_insights": ["ok"], "pred_summary": ""}
            result = generate(
                system="agentpoirot-official", benchmark_dir=root, case_id="flag-11",
                run_dir=root / "run", model="deepseek-flash", max_layers=3,
                questions=2, runner=fake_runner,
            )
        self.assertEqual(result["pred_insights"], ["ok"])
        self.assertEqual(calls[0]["goal"], "test goal")

    def test_split_defaults_to_contaminated_development_data(self):
        args = build_parser().parse_args([
            "--system", "agentpoirot-official", "--benchmark-dir", ".", "--case", "11",
        ])
        self.assertEqual(args.split, "dev-contaminated")

    def test_uncontrolled_split_name_is_rejected(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args([
                "--system", "agentpoirot-official", "--benchmark-dir", ".", "--case", "11",
                "--split", "test",
            ])


if __name__ == "__main__":
    unittest.main()
