import unittest

import numpy as np

from steering.runtime import SteeredPhase2Runtime
from steering.schemas import Phase1Action


class _FakeBackend:
    metadata = {"mode": "evo"}

    def predict(self, obs):
        del obs
        return Phase1Action(
            actions=np.asarray(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                ],
                dtype=np.float32,
            ),
            mode="evo",
        )


class _FakePhase2Policy:
    metadata = {}

    def __init__(self):
        self.phase1_action = None
        self.phase2_steps = None

    def infer_direct(self, obs, *, noise, sample_kwargs):
        del obs, noise, sample_kwargs
        return {"actions": np.zeros((2, 3), dtype=np.float32)}

    def refine(self, obs, *, phase1_action, phase2_steps, noise, sample_kwargs):
        del obs, noise, sample_kwargs
        self.phase1_action = phase1_action
        self.phase2_steps = phase2_steps
        return {"actions": phase1_action.actions}


class _FakeConfig:
    enable_by_default = False


class SteeredPhase2RuntimeTest(unittest.TestCase):
    def test_enabled_steerer_adds_hardcode_phase1_actions_to_backend_actions(self):
        phase2_policy = _FakePhase2Policy()
        runtime = SteeredPhase2Runtime(
            backend=_FakeBackend(),
            phase2_policy=phase2_policy,
            config=_FakeConfig(),
            enable_keys=("enable_steerer",),
        )

        result = runtime.infer(
            {},
            sample_kwargs={
                "enable_steerer": True,
                "phase2_steps": 3.0,
                "phase1_actions": np.asarray(
                    [
                        [0.5, 0.0, -0.5, 7.0],
                        [1.5, 1.0, -1.5, 8.0],
                    ],
                    dtype=np.float32,
                ),
            },
        )

        expected = np.asarray(
            [
                [1.5, 2.0, 2.5, 7.0],
                [5.5, 6.0, 4.5, 8.0],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(result["actions"], expected)
        self.assertEqual(phase2_policy.phase2_steps, 3.0)
        self.assertTrue(phase2_policy.phase1_action.metadata["combined_with_hardcode_phase1"])

    def test_metadata_marks_backend_steerer_enabled(self):
        runtime = SteeredPhase2Runtime(
            backend=_FakeBackend(),
            phase2_policy=_FakePhase2Policy(),
            config=_FakeConfig(),
            enable_keys=("enable_steerer",),
        )

        metadata = runtime.metadata

        self.assertTrue(metadata["steerer"]["enabled"])
        self.assertTrue(metadata["evo_steerer"]["enabled"])
        self.assertTrue(metadata["evo1_steerer"]["enabled"])


if __name__ == "__main__":
    unittest.main()
