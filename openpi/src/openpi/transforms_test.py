import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

import openpi.models.tokenizer as _tokenizer
import openpi.transforms as _transforms


def test_repack_transform():
    transform = _transforms.RepackTransform(
        structure={
            "a": {"b": "b/c"},
            "d": "e/f",
        }
    )
    item = {"b": {"c": 1}, "e": {"f": 2}}
    assert transform(item) == {"a": {"b": 1}, "d": 2}


def test_delta_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.DeltaActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 2, 5], [5, 4, 7]]))


def test_delta_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.DeltaActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.DeltaActions(mask=[True, False])
    assert transform(item) is item


def test_absolute_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.AbsoluteActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 6, 5], [5, 8, 7]]))


def test_absolute_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.AbsoluteActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.AbsoluteActions(mask=[True, False])
    assert transform(item) is item


def test_flexiv_state_delta_actions():
    state = np.zeros((3, 8), dtype=np.float32)
    state[:, 0] = [0.0, 0.1, 0.3]
    state[:, 3:7] = R.from_euler("z", np.array([0.0, 0.1, 0.3])[:, None]).as_quat().astype(np.float32)
    state[:, 7] = [0.01, 0.02, 0.03]
    item = {
        "observation/state": state,
        "actions": np.ones((2, 7), dtype=np.float32),
    }

    transformed = _transforms.FlexivStateDeltaActions()(item)

    assert transformed["observation/state"].shape == (8,)
    assert transformed["actions"].shape == (2, 7)
    np.testing.assert_allclose(transformed["actions"][:, :3], [[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    np.testing.assert_allclose(transformed["actions"][:, 3:6], [[0.0, 0.0, 0.1], [0.0, 0.0, 0.2]], atol=1e-6)
    np.testing.assert_allclose(transformed["actions"][:, 6], [0.02, 0.03])


@pytest.mark.parametrize("with_actions", [False, True])
def test_flexiv_state_delta_actions_inference_noop(with_actions):
    state = np.array([0, 0, 0, 0, 0, 0, 1, 0.02], dtype=np.float32)
    item = {"observation/state": state}
    if with_actions:
        item["actions"] = np.ones((2, 7), dtype=np.float32)
    result = _transforms.FlexivStateDeltaActions()(item)
    assert result is item
    assert result["observation/state"] is state
    if with_actions:
        np.testing.assert_array_equal(result["actions"], np.ones((2, 7)))


@pytest.mark.parametrize("shape", [(7,), (1, 8), (3, 7), (2, 2, 8)])
def test_flexiv_state_delta_actions_rejects_malformed_training_state(shape):
    with pytest.raises(ValueError, match="Expected observation/state"):
        _transforms.FlexivStateDeltaActions()({
            "observation/state": np.zeros(shape, dtype=np.float32),
            "actions": np.ones((2, 7), dtype=np.float32),
        })


def test_make_bool_mask():
    assert _transforms.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert _transforms.make_bool_mask(2, 0, 2) == (True, True, True, True)


def test_tokenize_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=12)
    transform = _transforms.TokenizePrompt(tokenizer)

    data = transform({"prompt": "Hello, world!"})

    tok_prompt, tok_mask = tokenizer.tokenize("Hello, world!")
    assert np.allclose(tok_prompt, data["tokenized_prompt"])
    assert np.allclose(tok_mask, data["tokenized_prompt_mask"])


def test_tokenize_no_prompt():
    transform = _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer())

    with pytest.raises(ValueError, match="Prompt is required"):
        transform({})


def test_transform_dict():
    # Rename and remove keys.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a/b": "a/c", "a/c": None}, input)
    assert output == {"a": {"c": 1}}

    # Raises and error since the renamed key conflicts with an existing key.
    with pytest.raises(ValueError, match="Key 'a/c' already exists in output"):
        _transforms.transform_dict({"a/b": "a/c"}, input)

    # Full match is required and so nothing will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a": None}, input)
    assert output == input

    # The regex matches the entire key and so the entire input will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a.+": None}, input)
    assert output == {}

    # Replace keys using backreferences. All leaves named 'c' are replaced with 'd'.
    input = {"a": {"b": 1, "c": 1}, "b": {"c": 2}}
    output = _transforms.transform_dict({"(.+)/c": r"\1/d"}, input)
    assert output == {"a": {"b": 1, "d": 1}, "b": {"d": 2}}


def test_extract_prompt_from_task():
    transform = _transforms.PromptFromLeRobotTask({1: "Hello, world!"})

    data = transform({"task_index": 1})
    assert data["prompt"] == "Hello, world!"

    with pytest.raises(ValueError, match="task_index=2 not found in task mapping"):
        transform({"task_index": 2})
