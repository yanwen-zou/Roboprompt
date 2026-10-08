"""Exercise launcher contracts without models, GPU initialization or robot access."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SERVER = 'scripts/realworld/eval/evo1/server/'
CLIENT = 'scripts/realworld/eval/evo1/client/'


class LauncherTest(unittest.TestCase):
    def run_script(self, name, values=None, extra=()):
        env = {k: v for k, v in os.environ.items() if k in ('PATH', 'HOME', 'LANG')}
        env.update(DRY_RUN='1', PYTHON_BIN='python3')
        env.update(values or {})
        return subprocess.run(['bash', str(ROOT / name), *extra], cwd='/tmp', env=env, text=True, capture_output=True)

    def test_all_server_presets_forward_assets_and_cli(self):
        values = dict(EVO1_CKPT='/tmp/evo checkpoint', PI05_CKPT='/tmp/pi checkpoint',
                      PI05_CONFIG='pi05_flexiv_bread', DP_CKPT='/tmp/dp.ckpt', DP_CONFIG='dp_config',
                      FASTWAM_CKPT='/tmp/fast.pt', FASTWAM_CONFIG='fast_config', FASTWAM_STATS='/tmp/stats.json',
                      PORT='9001', FRS='true')
        for path in (ROOT / SERVER).glob('*.sh'):
            with self.subTest(path=path.name):
                result = self.run_script(SERVER + path.name, values, ('--policy.steps=7',))
                self.assertEqual(result.returncode, 0, result.stderr)
                args = shlex.split(result.stdout)
                self.assertIn('--steerer.ckpt=/tmp/evo checkpoint', args)
                self.assertIn('--port=9001', args)
                self.assertIn('--policy.steps=7', args)
                self.assertIn('--frs', args)

    def test_missing_checkpoint_fails_before_python(self):
        result = self.run_script(SERVER + 'steer_server_openpi.sh')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('EVO1_CKPT', result.stderr)
        self.assertFalse(result.stdout)

    def test_nonexistent_checkpoint_fails_on_real_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_script(SERVER + 'steer_server_openpi.sh',
                                     dict(DRY_RUN='0', EVO1_CKPT=directory + '/missing'))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('does not exist', result.stderr)

    def test_clients_honor_host_and_interpreter(self):
        for path in (ROOT / CLIENT).glob('*.sh'):
            with self.subTest(path=path.name):
                result = self.run_script(CLIENT + path.name, dict(POLICY_HOST='policy.example', POLICY_PORT='9010'))
                self.assertEqual(result.returncode, 0, result.stderr)
                args = shlex.split(result.stdout)
                self.assertEqual(args[0], 'python3')
                self.assertEqual(args[args.index('--args.host') + 1], 'policy.example')
                self.assertEqual(args[args.index('--args.port') + 1], '9010')

    def test_collection_dry_run_never_imports_sdk(self):
        for task in ('bread', 'cup', 'maze', 'bean_T', 'noise', 'toy'):
            result = self.run_script(f'scripts/realworld/collect_demos/{task}.sh')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('hardware/record.py', result.stdout)


if __name__ == '__main__':
    unittest.main()
