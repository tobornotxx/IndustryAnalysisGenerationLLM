import unittest

from run_on_benchmark.split_registry import assigned_split, assert_case_split, split_registry_sha256


class SplitRegistryTests(unittest.TestCase):
    def test_source_membership_is_frozen(self):
        self.assertEqual(assigned_split("insightbench-overhaul", "flag-1"), "source-train")
        self.assertEqual(assigned_split("insightbench-overhaul", "flag-11"), "source-valid")
        self.assertEqual(assigned_split("insightbench-overhaul", "flag-20"), "source-test")

    def test_target_range_is_expanded_without_listing_every_case(self):
        self.assertEqual(assigned_split("insighteval-official", "insighteval-100"), "target-test")

    def test_mislabeled_case_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "frozen as source-test"):
            assert_case_split("insightbench-overhaul", "flag-13", "source-valid")

    def test_registry_has_stable_hash(self):
        self.assertEqual(len(split_registry_sha256()), 64)


if __name__ == "__main__":
    unittest.main()
