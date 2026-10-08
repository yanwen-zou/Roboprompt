import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_robocasa_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(16),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
        "prompt_images": {
            "prompt_0": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        },
        "prompt_image_masks": {
            "prompt_0": np.True_,
        },
        "prompt_global_motion": np.asarray([-1.0, 0.0, 1.0], dtype=np.float32),
        "prompt_global_motion_mask": np.True_,
        "prompt_local_motion": np.random.randn(3).astype(np.float32),
        "prompt_local_motion_mask": np.True_,
        "prompt_2d_drag": np.asarray([0.2, -0.1], dtype=np.float32),
        "prompt_2d_drag_mask": np.True_,
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RobocasaInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType = _model.ModelType.PI0
    include_prompt_images: bool = True

    def __call__(self, data: dict) -> dict:
        # We only mask padding for pi0 model, not pi0-FAST. Do not change this for your own dataset.
        mask_padding = self.model_type == _model.ModelType.PI0

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Mask any non-existent images with False (if ``mask_padding`` is True).
                "right_wrist_0_rgb": np.False_ if mask_padding else np.True_,
            },
        }

        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        if self.include_prompt_images and "prompt_images" in data:
            inputs["prompt_images"] = {
                name: _parse_image(image) for name, image in data["prompt_images"].items()
            }

        if self.include_prompt_images and "prompt_image_masks" in data:
            inputs["prompt_image_masks"] = {
                name: np.asarray(mask, dtype=np.bool_) for name, mask in data["prompt_image_masks"].items()
            }

        prompt_aliases = {
            "prompt_local_motion": "prompt_wrist_relative_action",
            "prompt_local_motion_mask": "prompt_wrist_relative_action_mask",
            "prompt_global_motion": "prompt_primitive_cmd",
            "prompt_global_motion_mask": "prompt_primitive_cmd_mask",
        }
        for key, legacy_key in prompt_aliases.items():
            if key in data:
                value = data[key]
            elif legacy_key in data:
                value = data[legacy_key]
            else:
                continue
            dtype = np.bool_ if key.endswith("_mask") else np.float32
            inputs[key] = np.asarray(value, dtype=dtype)

        if "prompt_2d_drag" in data:
            inputs["prompt_2d_drag"] = np.asarray(data["prompt_2d_drag"], dtype=np.float32)

        if "prompt_2d_drag_mask" in data:
            inputs["prompt_2d_drag_mask"] = np.asarray(data["prompt_2d_drag_mask"], dtype=np.bool_)

        for key in ("prompt_traj_2d", "prompt_point_2d"):
            if key in data:
                inputs[key] = np.asarray(data[key], dtype=np.float32)

        for key in ("prompt_traj_mask", "prompt_point_mask"):
            if key in data:
                inputs[key] = np.asarray(data[key], dtype=np.bool_)

        for key in (
            "prompt_world_to_camera",
            "prompt_camera_resolution",
            "prompt_tcp_world_pos",
            "prompt_tcp_rot",
            "prompt_base_rot",
            "prompt_action_q01",
            "prompt_action_q99",
            "prompt_wrist_action_q01",
            "prompt_wrist_action_q99",
        ):
            if key in data:
                inputs[key] = np.asarray(data[key], dtype=np.float32)

        return inputs


@dataclasses.dataclass(frozen=True)
class RobocasaOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][:, :12])}
