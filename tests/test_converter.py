"""Synthetic format checks; no competition motion data is used."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "convert_bones_csv_to_deploy.py"
spec = importlib.util.spec_from_file_location("converter", SCRIPT)
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class ConverterTests(unittest.TestCase):
    def make_data(self):
        frames = np.arange(121, dtype=float)
        values = np.zeros((121, 36))
        values[:, 0] = frames
        values[:, 1] = frames  # root X in cm, 120 Hz
        values[:, 7] = 90.0  # first joint in degrees
        return values

    def check_output(self, input_header):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            csv_path = base / "sample.csv"
            values = self.make_data()
            if input_header:
                columns = ["Frame", "root_translateX", "root_translateY", "root_translateZ",
                           "root_rotateX", "root_rotateY", "root_rotateZ"] + [f"joint_{i}_dof" for i in range(29)]
                pd.DataFrame(values, columns=columns).to_csv(csv_path, index=False)
            else:
                np.savetxt(csv_path, values, delimiter=",")
            converter.process_single_csv(csv_path, base / "out", 120)
            result = base / "out" / "sample"
            pos = pd.read_csv(result / "joint_pos.csv").to_numpy()
            vel = pd.read_csv(result / "joint_vel.csv").to_numpy()
            body = pd.read_csv(result / "body_pos.csv").to_numpy()
            quat = pd.read_csv(result / "body_quat.csv").to_numpy()
            self.assertEqual(pos.shape, (51, 29))
            self.assertEqual(vel.shape, (51, 29))
            self.assertEqual(body.shape, (51, 3))
            self.assertEqual(quat.shape, (51, 4))
            self.assertAlmostEqual(body[-1, 0], 1.2)
            self.assertAlmostEqual(pos[0, converter.MJ_TO_IL[0]], np.pi / 2)
            np.testing.assert_allclose(vel, 0, atol=1e-10)
            np.testing.assert_allclose(quat[:, 0], 1, atol=1e-10)

    def test_headered_csv(self):
        self.check_output(True)

    def test_headerless_csv(self):
        self.check_output(False)

    def test_rejects_bad_input_without_output(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            csv_path = base / "short.csv"
            np.savetxt(csv_path, np.zeros((1, 36)), delimiter=",")
            with self.assertRaises(ValueError):
                converter.process_single_csv(csv_path, base / "out", 120)
            self.assertFalse((base / "out").exists())
            values = self.make_data()
            values[5, 8] = np.nan
            np.savetxt(csv_path, values, delimiter=",")
            with self.assertRaises(ValueError):
                converter.process_single_csv(csv_path, base / "out", 120)
            self.assertFalse((base / "out").exists())


if __name__ == "__main__":
    unittest.main()
