from pathlib import Path
import unittest

from autocamtracker.core.gid_loss_benchmark import load_gid_loss_benchmark


class GIDLossBenchmarkTests(unittest.TestCase):
    def test_manifest_defines_required_loss_scenarios(self) -> None:
        benchmark = load_gid_loss_benchmark(Path("evaluation/gid_loss_scenarios.json"))

        self.assertEqual(
            [scenario.scenario_id for scenario in benchmark.scenarios],
            ["occlusion", "crossing_traffic", "fast_lateral", "exit_reenter"],
        )
        self.assertIn("gid_lock_rate_min", benchmark.metrics)
        self.assertIn("median_reacquire_frames_max", benchmark.metrics)
        self.assertEqual(benchmark.summary()["scenario_count"], 4)


if __name__ == "__main__":
    unittest.main()
