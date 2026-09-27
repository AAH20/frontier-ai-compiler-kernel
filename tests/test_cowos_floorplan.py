"""Unit tests for CoWoS Chiplet Floorplanning Solver."""

import unittest
from frontier_ai_compiler_kernel.core.cowos_chiplet_floorplanner import CowosChipletFloorplanner
from frontier_ai_compiler_kernel.core.models import ChipletDie


class TestCowosFloorplan(unittest.TestCase):
    def setUp(self):
        self.floorplanner = CowosChipletFloorplanner()
        self.dies = [
            ChipletDie("gpu0", "GPU_CORE", 20.0, 20.0, 500.0),
            ChipletDie("gpu1", "GPU_CORE", 20.0, 20.0, 500.0),
            ChipletDie("hbm0", "HBM3E", 10.0, 10.0, 30.0),
            ChipletDie("hbm1", "HBM3E", 10.0, 10.0, 30.0),
        ]
        self.connectivity = {
            ("gpu0", "gpu1"): 10.0,
            ("gpu0", "hbm0"): 5.0,
            ("gpu1", "hbm1"): 5.0,
        }

    def test_thermal_safety_and_placement(self):
        report = self.floorplanner.floorplan_chiplets(self.dies, self.connectivity)

        self.assertEqual(len(report.placements), 4)
        self.assertTrue(report.is_thermally_safe)
        self.assertLessEqual(report.peak_die_temperature_c, 95.0)


if __name__ == "__main__":
    unittest.main()
