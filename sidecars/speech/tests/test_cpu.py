import os
import unittest
from unittest import mock

from speech_app.cpu import choose_cpu_plan


class CpuPolicyTests(unittest.TestCase):
    def test_auto_policy_scales_and_keeps_system_headroom(self):
        cases = [
            (1, 1, 1),
            (2, 1, 1),
            (4, 2, 2),
            (8, 4, 4),
            (16, 8, 8),
            (32, 16, 12),
            (64, 32, 12),
        ]
        for logical, physical, expected in cases:
            with self.subTest(logical=logical, physical=physical):
                plan = choose_cpu_plan(logical, physical, override="")
                self.assertEqual(plan.inference_threads, expected)
                self.assertGreaterEqual(plan.reserved_logical, 0)

    def test_override_is_clamped_to_available_logical_cpus(self):
        self.assertEqual(
            choose_cpu_plan(8, 4, override="99").inference_threads,
            8,
        )
        self.assertEqual(choose_cpu_plan(8, 4, override="2").source, "override")

    def test_invalid_override_falls_back_to_auto(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            self.assertEqual(
                choose_cpu_plan(16, 8, override="not-a-number").source,
                "auto",
            )


if __name__ == "__main__":
    unittest.main()
