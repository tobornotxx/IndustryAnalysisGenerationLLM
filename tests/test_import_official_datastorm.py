import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from run_on_benchmark.import_official_datastorm import OFFICIAL_COMMIT, import_prediction


class OfficialDataStormImportTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        official = root / "official"
        source = official / "results" / "insight_bench" / "scores"
        source.mkdir(parents=True)
        (source / "pred_gt_11.json").write_text(json.dumps({
            "dataset_id": 11,
            "metadata": {"goal": "find trends"},
            "predicted_insights": ["one", "two"],
            "llm_predicted_summary": "summary",
            "score_insights": 0.6,
        }), encoding="utf-8")
        benchmark = root / "benchmark"
        gt = benchmark / "data" / "notebooks"
        gt.mkdir(parents=True)
        (gt / "flag-11.json").write_text("{}", encoding="utf-8")
        return official, benchmark

    @patch("run_on_benchmark.import_official_datastorm._git_head", return_value=OFFICIAL_COMMIT)
    def test_imports_without_generation_and_preserves_provenance(self, _head):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            official, benchmark = self._fixture(root)
            run = import_prediction(
                official_repo=official, benchmark_dir=benchmark, out_root=root / "out",
                experiment_id="published", system_id="datastorm-official-published",
                case_number=11,
            )
            prediction = json.loads((run / "prediction.json").read_text(encoding="utf-8"))
            manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
            scores = json.loads((run / "published_scores.json").read_text(encoding="utf-8"))
        self.assertEqual(prediction["pred_insights"], ["one", "two"])
        self.assertEqual(prediction["artifact_origin"], "author-published")
        self.assertEqual(manifest["source"]["commit"], OFFICIAL_COMMIT)
        self.assertEqual(scores["scores"]["score_insights"], 0.6)

    @patch("run_on_benchmark.import_official_datastorm._git_head", return_value="wrong")
    def test_rejects_unpinned_official_checkout(self, _head):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            official, benchmark = self._fixture(root)
            with self.assertRaisesRegex(ValueError, "must be pinned"):
                import_prediction(
                    official_repo=official, benchmark_dir=benchmark, out_root=root / "out",
                    experiment_id="published", system_id="datastorm-official-published",
                    case_number=11,
                )


if __name__ == "__main__":
    unittest.main()
