import tempfile
import unittest
from pathlib import Path

from run_on_benchmark.run_baseline_matrix import plan_tasks


class BaselineMatrixTests(unittest.TestCase):
    def test_plans_all_system_case_repeat_combinations(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks = plan_tasks(
                out_root=Path(tmp), experiment="round1",
                systems=["legacy-custom-python", "agentpoirot-upstream-local"],
                cases=["flag-11", "flag-12"], agent_runs=2,
            )
        self.assertEqual(len(tasks), 8)
        self.assertTrue(all(task["existing_status"] is None for task in tasks))

    def test_rejects_unknown_system(self):
        with self.assertRaises(ValueError):
            plan_tasks(
                out_root=Path("results"), experiment="round1",
                systems=["invented"], cases=["flag-11"], agent_runs=1,
            )


if __name__ == "__main__":
    unittest.main()
