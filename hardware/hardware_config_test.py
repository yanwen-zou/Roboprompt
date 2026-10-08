import json
import os
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / 'hardware/my_device/macros.py'
REFERENCE = ROOT / 'hardware/config/flexiv.reference.json'


class HardwareConfigTest(unittest.TestCase):
    def test_configuration_is_explicit(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'ROBOPROMPT_HARDWARE_CONFIG'):
                runpy.run_path(str(MODULE))

    def test_reference_values_are_preserved(self):
        with patch.dict(os.environ, ROBOPROMPT_HARDWARE_CONFIG=str(REFERENCE)):
            config = runpy.run_path(str(MODULE))
        self.assertEqual(config['CAM_SERIAL'], json.loads(REFERENCE.read_text())['CAM_SERIAL'])
        self.assertEqual(config['E2H_CAM_T'].shape, (4, 4))
        self.assertEqual(config['HOME_JOINT_DEG'].shape, (7,))

    def test_invalid_calibration_is_rejected(self):
        for key, value in [('CAM_SERIAL', ['duplicate', 'duplicate']), ('E2H_CAM_T', [[1]]),
                           ('HOME_JOINT_DEG', [float('nan')] * 7), ('ROBOT_SN', '')]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                config = json.loads(REFERENCE.read_text())
                config[key] = value
                path = Path(directory) / 'config.json'
                path.write_text(json.dumps(config))
                with patch.dict(os.environ, ROBOPROMPT_HARDWARE_CONFIG=str(path)):
                    with self.assertRaises(ValueError):
                        runpy.run_path(str(MODULE))


if __name__ == '__main__':
    unittest.main()
