from types import SimpleNamespace

import numpy as np

from scripts.utils.interactive_prompt import InteractivePromptState
from scripts.utils.interactive_prompt import denoise_step_from_sample_kwargs
from scripts.utils.interactive_prompt import phase2_prompt_mem_step
from scripts.utils.interactive_prompt import policy_inference_steps_from_metadata


def test_ui_short_term_uses_selected_step_then_returns_to_max_steps() -> None:
    prompt_state = InteractivePromptState()
    prompt_state.update(
        {
            "prompt_2d_drag": np.zeros((2,), dtype=np.float32),
            "prompt_2d_drag_mask": np.bool_(True),
            "sample_kwargs": {"phase2_steps": 1.0, "num_steps": 10.0},
            "prompt_effect_mode": "short_term",
        }
    )

    assert prompt_state.build_sample_kwargs()["phase2_steps"] == 1.0

    prompt_state.add_prompt_mem()
    assert prompt_state.build_sample_kwargs()["phase2_steps"] == 10.0

    prompt_state.add_prompt_mem()
    assert prompt_state.build_sample_kwargs()["phase2_steps"] == 10.0


def test_ui_long_term_increments_phase2_steps_across_action_chunks() -> None:
    prompt_state = InteractivePromptState()
    prompt_state.update(
        {
            "prompt_2d_drag": np.zeros((2,), dtype=np.float32),
            "prompt_2d_drag_mask": np.bool_(False),
            "sample_kwargs": {"phase2_steps": 1.0, "num_steps": 10.0},
            "prompt_effect_mode": "long_term",
            "interactive_prompt_result": SimpleNamespace(
                draw_ops=[
                    {
                        "type": "point",
                        "point_hw": [10.0, 20.0],
                        "source": "right_click_point",
                    }
                ]
            ),
        }
    )

    assert prompt_state.build_sample_kwargs()["phase2_steps"] == 1.0

    prompt_state.add_prompt_mem()
    assert prompt_state.build_sample_kwargs()["phase2_steps"] == 1.5

    prompt_state.add_prompt_mem()
    assert prompt_state.build_sample_kwargs()["phase2_steps"] == 2.0


def test_ui_long_term_increments_without_motion_prompt() -> None:
    prompt_state = InteractivePromptState()
    prompt_state.update(
        {
            "sample_kwargs": {"phase2_steps": 0.0, "num_steps": 10.0},
            "prompt_effect_mode": "long_term",
        }
    )

    prompt_state.add_prompt_mem()
    assert prompt_state.build_sample_kwargs()["phase2_steps"] == 0.5


def test_ui_long_term_increment_scales_with_policy_inference_steps() -> None:
    prompt_state = InteractivePromptState()
    prompt_state.update(
        {
            "sample_kwargs": {"phase2_steps": 1.0},
            "prompt_effect_mode": "long_term",
        }
    )

    prompt_state.add_prompt_mem()
    assert prompt_state.build_sample_kwargs(policy_inference_steps=20.0)["phase2_steps"] == 2.0


def test_ui_long_term_adds_phase2_steps_and_random_noise_ratio() -> None:
    prompt_state = InteractivePromptState()
    prompt_state.update(
        {
            "sample_kwargs": {"phase2_steps": 1.0},
            "prompt_effect_mode": "long_term",
        }
    )

    prompt_state.add_prompt_mem()
    sample_kwargs = prompt_state.build_sample_kwargs(policy_inference_steps=20.0)
    assert sample_kwargs["phase2_steps"] == 2.0
    assert sample_kwargs["random_noise_ratio"] == 0.2

    prompt_state.add_prompt_mem()
    sample_kwargs = prompt_state.build_sample_kwargs(policy_inference_steps=20.0)
    assert sample_kwargs["phase2_steps"] == 3.0
    assert sample_kwargs["random_noise_ratio"] == 0.4


def test_ui_long_term_can_disable_random_noise_ratio() -> None:
    prompt_state = InteractivePromptState()
    prompt_state.update(
        {
            "sample_kwargs": {"phase2_steps": 1.0},
            "prompt_effect_mode": "long_term",
        }
    )

    prompt_state.add_prompt_mem()
    sample_kwargs = prompt_state.build_sample_kwargs(
        policy_inference_steps=20.0,
        random_noise_ratio_step=0.0,
    )
    assert sample_kwargs["phase2_steps"] == 2.0
    assert "random_noise_ratio" not in sample_kwargs


def test_ui_long_term_uses_custom_random_noise_ratio_step() -> None:
    prompt_state = InteractivePromptState()
    prompt_state.update(
        {
            "sample_kwargs": {"phase2_steps": 1.0},
            "prompt_effect_mode": "long_term",
        }
    )

    prompt_state.add_prompt_mem()
    sample_kwargs = prompt_state.build_sample_kwargs(
        policy_inference_steps=20.0,
        phase2_policy_type="fastwam",
        random_noise_ratio_step=0.1,
    )
    assert sample_kwargs["phase2_steps"] == 2.0
    assert sample_kwargs["random_noise_ratio"] == 0.1


def test_ui_long_term_lifts_zero_phase2_base() -> None:
    prompt_state = InteractivePromptState()
    prompt_state.update(
        {
            "sample_kwargs": {"phase2_steps": 0.0},
            "prompt_effect_mode": "long_term",
        }
    )

    prompt_state.add_prompt_mem()
    sample_kwargs = prompt_state.build_sample_kwargs(
        policy_inference_steps=20.0,
        phase2_policy_type="fastwam",
    )
    assert sample_kwargs["phase2_steps"] == 1.0
    assert sample_kwargs["random_noise_ratio"] == 0.2
    assert (
        denoise_step_from_sample_kwargs(
            sample_kwargs,
            policy_inference_steps=20.0,
            phase2_policy_type="fastwam",
        )
        == 1.0
    )


def test_ui_short_term_uses_server_policy_inference_steps_when_payload_has_no_num_steps() -> None:
    prompt_state = InteractivePromptState()
    prompt_state.update(
        {
            "sample_kwargs": {"phase2_steps": 1.0},
            "prompt_effect_mode": "short_term",
        }
    )

    prompt_state.add_prompt_mem()
    assert prompt_state.build_sample_kwargs(policy_inference_steps=20.0)["phase2_steps"] == 20.0


def test_ui_can_emit_num_steps_for_openpi() -> None:
    prompt_state = InteractivePromptState()
    prompt_state.update(
        {
            "sample_kwargs": {"phase2_steps": 1.0},
            "prompt_effect_mode": "long_term",
        }
    )

    sample_kwargs = prompt_state.build_sample_kwargs(policy_inference_steps=20.0, emit_num_steps=True)
    assert sample_kwargs["num_steps"] == 20.0


def test_policy_inference_steps_from_metadata_prefers_server_max_phase2_steps() -> None:
    assert policy_inference_steps_from_metadata({"max_phase2_steps": 20, "num_steps": 10}) == 20.0


def test_denoise_step_from_random_noise_ratio_uses_server_policy_steps_without_num_steps() -> None:
    sample_kwargs = {
        "phase2_steps": 20.0,
        "random_noise_ratio": 1.0,
    }

    assert denoise_step_from_sample_kwargs(sample_kwargs, policy_inference_steps=20.0) == 20.0


def test_denoise_step_from_random_noise_ratio_keeps_base_phase2_steps() -> None:
    sample_kwargs = {
        "phase2_steps": 2.0,
        "random_noise_ratio": 0.2,
    }

    denoise_step = denoise_step_from_sample_kwargs(sample_kwargs, policy_inference_steps=20.0)
    assert denoise_step == 2.0


def test_fastwam_denoise_step_from_random_noise_ratio_keeps_base_phase2_steps() -> None:
    sample_kwargs = {
        "phase2_steps": 1.0,
        "random_noise_ratio": 0.4,
    }

    denoise_step = denoise_step_from_sample_kwargs(
        sample_kwargs,
        policy_inference_steps=20.0,
        phase2_policy_type="fastwam",
    )
    assert denoise_step == 1.0



def test_phase2_prompt_mem_step_is_five_percent_of_infer_steps() -> None:
    assert phase2_prompt_mem_step(10.0) == 0.5
    assert phase2_prompt_mem_step(20.0) == 1.0
