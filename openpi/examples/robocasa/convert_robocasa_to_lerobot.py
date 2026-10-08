"""
Minimal example script for converting a dataset to LeRobot format.

We use the Libero dataset (stored in RLDS) for this example, but it can be easily
modified for any other data you have saved in a custom format.

Usage:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

Note: to run the script, you need to install tensorflow_datasets:
`uv pip install tensorflow tensorflow_datasets`

You can download the raw Libero datasets from https://huggingface.co/datasets/openvla/modified_libero_rlds
The resulting dataset will get saved to the $LEROBOT_HOME directory.
Running this conversion script will take approximately 30 minutes.
"""

import shutil

# from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro
import h5py
import numpy as np
import json
import os
from tqdm import tqdm

# REPO_NAME = "robocasa/atomic_5tasks_mg"  # Name of the output dataset, also used for the Hugging Face Hub
# RAW_DATASET_PATHS = [
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/CloseStandMixerHead/mg/demo/2025-07-12-14-59-28/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/PnPCounterToSink/mg/demo/2025-07-12-15-02-36/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/StartCoffeeMachine/mg/demo/2025-07-11-18-31-31/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/TurnOffStove/mg/demo/2025-07-11-18-29-00/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/OpenDishwasher/mg/demo/2025-07-12-15-00-12/demo_im128.hdf5',
# ]

# REPO_NAME = "robocasa/atomic_human"
# RAW_DATASET_PATHS = [
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/AdjustToasterOvenTemperature/20250714/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/CloseCabinet/20250711/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/CloseDishwasher/20250711/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/CloseDrawer/20250714/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/CloseElectricKettleLid/20250712/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/CloseFridge/20250711/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/CloseMicrowave/20250712/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/CloseOven/20250711/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/CloseStandMixerHead/20250711/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/CloseToasterOvenDoor/20250714/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/CoffeeServeMug/20250711/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/CoffeeSetupMug/20250711/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/OpenCabinet/20250711/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/OpenDishwasher/20250711/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/OpenDrawer/20250714/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/OpenElectricKettleLid/20250712/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/OpenFridge/20250711/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/OpenMicrowave/20250711/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/OpenOven/20250711/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/OpenStandMixerHead/20250711/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/OpenToasterOvenDoor/20250714/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/PnPCabinetToCounter/20250711/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/PnPCounterToCabinet/20250712/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/PnPCounterToMicrowave/20250712/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/PnPCounterToOven/20250714/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/PnPCounterToSink/20250711/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/PnPCounterToStandMixer/20250711/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/PnPCounterToStove/20250711/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/PnPCounterToToasterOven/20250714/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/PnPMicrowaveToCounter/20250711/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/PnPSinkToCounter/20250711/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/PnPStoveToCounter/20250711/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/PnPToasterOvenToCounter/20250714/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/PnPToasterToCounter/20250714/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/StartCoffeeMachine/20250711/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/TurnOffMicrowave/20250711/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/TurnOffSinkFaucet/20250711/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/TurnOffStove/20250711/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/TurnOnElectricKettle/20250712/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/TurnOnMicrowave/20250711/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/TurnOnSinkFaucet/20250711/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/TurnOnStove/20250712/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/TurnOnToaster/20250714/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/TurnOnToasterOven/20250714/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/TurnSinkSpout/20250711/demo_im128.hdf5',
# ]

# REPO_NAME = "robocasa/PnPCounterToCabinet_mg_18k"
# RAW_DATASET_PATHS = [
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/PnPCounterToCabinet/mg/demo/2025-07-18-15-05-32/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/PnPCounterToCabinet/mg/demo/2025-07-18-20-17-06/demo_im128.hdf5',
# ]

# REPO_NAME = "robocasa/PnPCounterToCabinet_mg_18k"
# RAW_DATASET_PATHS = [
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/PnPCounterToCabinet/mg/demo/2025-07-18-15-05-32/demo_im128.hdf5',
#     '/mnt/nfs_client/robocasa/datasets/v0.5/train/atomic/PnPCounterToCabinet/mg/demo/2025-07-18-20-17-06/demo_im128.hdf5',
# ]

REPO_NAME = "robocasa365/CloseStandMixerHead_human"
RAW_DATASET_PATHS = [
    '/mnt/amlfs-01/shared/robocasa_benchmark/robocasa-datasets/v0.5/train/atomic/CloseStandMixerHead/20250711/demo_im128.hdf5',
]

# LEROBOT_HOME = "/home/soroush/.cache/huggingface/lerobot"
LEROBOT_HOME = "/mnt/amlfs-01/shared/robocasa_benchmark/misc/lerobot"

def main(*, push_to_hub: bool = False):
    # Clean up any existing dataset in the output directory
    output_path = os.path.join(LEROBOT_HOME, REPO_NAME)
    if os.path.exists(output_path):
    # if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=20,
        features={
            "image": {
                "dtype": "image",
                "shape": (128, 128, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (128, 128, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float64",
                "shape": (16,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float64",
                "shape": (12,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw Libero datasets and write episodes to the LeRobot dataset
    # You can modify this for your own data format
   
    for raw_dataset_path in RAW_DATASET_PATHS:
        raw_dataset = h5py.File(raw_dataset_path, "r")
        demos = raw_dataset["data"].keys()
        # demos = list(demos)[:10]
        for demo in tqdm(demos):
            demo_length = len(raw_dataset["data"][demo]["actions"])
            demo_data = raw_dataset["data"][demo]
            
            images = demo_data["obs"]["robot0_agentview_left_image"][:]
            wrist_images = demo_data["obs"]["robot0_eye_in_hand_image"][:]
            states = np.concatenate(
                (
                    demo_data["obs"]["robot0_base_to_eef_pos"][:],
                    demo_data["obs"]["robot0_base_to_eef_quat"][:],
                    demo_data["obs"]["robot0_base_pos"][:],
                    demo_data["obs"]["robot0_base_quat"][:],
                    demo_data["obs"]["robot0_gripper_qpos"][:],
                ),
                axis=1,
            )
            actions = demo_data["actions"][:]

            ep_meta = demo_data.attrs["ep_meta"]
            ep_meta = json.loads(ep_meta)
            lang = ep_meta["lang"]

            for i in range(demo_length):
                dataset.add_frame(
                    {
                        "image": images[i],
                        "wrist_image": wrist_images[i],
                        "state": states[i],
                        "actions": actions[i],
                        "task": lang,
                    }
                )

            dataset.save_episode() #task=lang)

    # Consolidate the dataset, skip computing stats since we will do that later
    # dataset.consolidate(run_compute_stats=False)

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["robocasa", "panda", "hdf5"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
