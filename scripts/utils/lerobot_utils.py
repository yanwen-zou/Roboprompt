"""LeRobot video encoding for real robot demonstration conversion.

Extracted from the formerly bundled RoboCasa utility.
"""

import shutil

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import write_info
from lerobot.common.datasets.video_utils import encode_video_frames


class LerobotDatasetWrapper(LeRobotDataset):
    """
    Wrapper class for creating LeRobotDataset. Class is needed so that we can override
    methods to get more control over video encoding parameters.
    """

    def encode_episode_videos(self, episode_index: int) -> dict:
        """
        Encode videos for a given episode index. Code is mostly copied from parent class
        but with modified video new encoding parameters.
        """
        video_paths = {}
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            video_paths[key] = str(video_path)
            if video_path.is_file():
                # Skip if video is already encoded. Could be the case when resuming data recording.
                continue
            img_dir = self._get_image_file_path(
                episode_index=episode_index, image_key=key, frame_index=0
            ).parent
            encode_video_frames(
                img_dir,
                video_path,
                self.fps,
                overwrite=True,
                vcodec="h264",
                pix_fmt="yuv420p",
                crf=23,
                g=None,
                fast_decode=0,
            )
            shutil.rmtree(img_dir)

        # Update video info (only needed when first episode is encoded since it reads from episode 0)
        if len(self.meta.video_keys) > 0 and episode_index == 0:
            self.meta.update_video_info()
            write_info(
                self.meta.info, self.meta.root
            )  # ensure video info always written properly
        return video_paths
