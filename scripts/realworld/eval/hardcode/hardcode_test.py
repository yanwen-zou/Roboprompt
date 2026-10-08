import numpy as np

from scripts.realworld.eval.hardcode import hardcode


def _identity_norm_metadata(action_dim: int = 32) -> dict:
    return {
        "action_norm_stats": {
            "min": np.full((action_dim,), -1.0, dtype=np.float32),
            "max": np.full((action_dim,), 1.0, dtype=np.float32),
        }
    }


def test_local_motion_is_rotated_from_wrist_to_base_frame() -> None:
    observation_state = np.asarray(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.4,
        ],
        dtype=np.float32,
    )

    phase1_actions = hardcode.build_phase1_action_chunk(
        prompt_payload={
            "prompt_local_motion": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            "prompt_local_motion_mask": np.bool_(True),
        },
        observation_state=observation_state,
        metadata=_identity_norm_metadata(),
        action_horizon=4,
    )

    assert phase1_actions is not None
    np.testing.assert_allclose(phase1_actions[:, :3], np.asarray([[-1.0, 0.0, 0.0]] * 4), atol=1e-6)
    np.testing.assert_allclose(phase1_actions[:, 6], np.asarray([0.4] * 4, dtype=np.float32))


def test_phase1_action_chunk_repeats_command_over_full_horizon() -> None:
    phase1_actions = hardcode.build_phase1_action_chunk(
        prompt_payload={
            "prompt_global_motion": np.asarray([0.2, 0.0, -0.3], dtype=np.float32),
            "prompt_global_motion_mask": np.bool_(True),
        },
        observation_state=np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -0.2], dtype=np.float32),
        metadata=_identity_norm_metadata(),
        action_horizon=3,
    )

    assert phase1_actions is not None
    np.testing.assert_allclose(phase1_actions[:, :3], np.asarray([[0.2, 0.0, -0.3]] * 3), atol=1e-6)
    np.testing.assert_allclose(phase1_actions[:, 6], np.asarray([-0.2] * 3, dtype=np.float32))


def test_has_2d_prompt_accepts_point_prompt_without_drag_vector() -> None:
    assert hardcode.has_2d_prompt(
        {
            "prompt_2d_drag_mask": np.bool_(False),
            "prompt_image_masks": {"prompt_0": np.bool_(True)},
        }
    )
