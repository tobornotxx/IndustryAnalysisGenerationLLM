import unittest

from run_on_benchmark.system_registry import canonical_system_id, system_record


class SystemRegistryTests(unittest.TestCase):
    def test_canonical_identity_is_available(self):
        record = system_record("datastorm-official-published")
        self.assertEqual(record["generation_mode"], "published-artifact")
        self.assertEqual(record["generation_model"], "gpt-5-2025-08-07")

    def test_retired_misleading_identity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "legacy-custom-python"):
            canonical_system_id("datastorm-reproduction")

    def test_alias_can_only_be_resolved_explicitly(self):
        self.assertEqual(
            canonical_system_id("datastorm-reproduction", allow_retired=True),
            "legacy-custom-python",
        )


if __name__ == "__main__":
    unittest.main()
