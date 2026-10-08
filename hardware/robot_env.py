import numpy as np
import torch
import time
import cv2
from PIL import Image
from scipy.spatial.transform import Rotation as R
import pygame
from torchvision.transforms import Compose, Resize, CenterCrop
from torchvision.transforms import InterpolationMode

from flexiv.robot import FlexivRobot, FlexivGripper
from my_device.camera import CameraD400
from my_device.keyboard import Keyboard
from my_device.sigma import Sigma7
from my_device.logitechG29_wheel import Controller
from my_device.macros import CAM_SERIAL, INTV


class RobotEnv:
    def __init__(
        self,
        camera_serial=CAM_SERIAL,
        img_shape=None,
        fps=10,
        disable_gripper_cmd=False,
        z_freeze_clutch_threshold=None,
    ):
        self.camera_serial = list(camera_serial) if camera_serial is not None else list(CAM_SERIAL)
        self.fps = fps
        self.img_shape = img_shape
        self.disable_gripper_cmd = bool(disable_gripper_cmd)
        self.z_freeze_clutch_threshold = z_freeze_clutch_threshold
        self.sigma_z_bias = 0.0
        self.z_locked = False

        # Initialize hardware components
        self.robot = FlexivRobot()
        self.sigma = Sigma7()
        pygame.init()
        self.controller = Controller(0)
        self.gripper = FlexivGripper(self.robot)
        self.zed = self._init_camera()
        self.keyboard = Keyboard()
        self.home_pose = self.robot.init_pose
        
        # Setup image processors
        BICUBIC = InterpolationMode.BICUBIC
        self.image_processor = Compose([
            Resize((img_shape[1]+8, img_shape[2]+8), interpolation=BICUBIC),
            CenterCrop((img_shape[1], img_shape[2]))
        ])
        
        # Keep track of throttle usage for human intervention
        self.last_throttle = False

    def _init_camera(self):
        if len(self.camera_serial) < 2:
            raise ValueError(
                f"Expected two camera serials for env/wrist cameras, got {self.camera_serial!r}."
            )
        return {
            # temporary change for active glass
            "env": CameraD400(self.camera_serial[0]),
            # "env": CameraD400(self.camera_serial[2]),
            "wrist": CameraD400(self.camera_serial[1]),
        }

    def close(self, timeout_s: float = 15.0) -> None:
        """Close resources."""
        for camera in self.zed.values():
            del camera
        self.robot.stop()
        self.sigma.close()
        pygame.quit()

    def _get_camera_frames(self):
        try:
            env_bgr, _ = self.zed["env"].get_data()
            wrist_bgr, _ = self.zed["wrist"].get_data()
        except Exception as exc:
            print(f"Failed to read images from cameras: {exc}")
            return None, None
        env_rgb = cv2.cvtColor(env_bgr, cv2.COLOR_BGR2RGB)
        wrist_rgb = cv2.cvtColor(wrist_bgr, cv2.COLOR_BGR2RGB)
        return env_rgb, wrist_rgb
        
    def reset_robot(self, random_init=False, random_init_pose=None):
        if random_init and random_init_pose is not None:
            self.robot.send_tcp_pose(random_init_pose)
        else:
            self.robot.send_tcp_pose(self.robot.init_pose)
        time.sleep(2)
        self.gripper.move(self.gripper.max_width)
        time.sleep(0.5)
        print("Reset!")

        self.sigma.reset()
        self.last_throttle = False
        self.sigma_z_bias = 0.0
        self.z_locked = False
        if random_init and random_init_pose is not None:
            random_p_drift = random_init_pose[:3] - self.robot.init_pose[:3]
            random_r_drift = R.from_quat(self.robot.init_pose[3:7], scalar_first=True).inv() * R.from_quat(random_init_pose[3:7], scalar_first=True)
            self.sigma.transform_from_robot(random_p_drift, random_r_drift)
        
        return self.get_robot_state()
    
    def get_robot_state(self):
        """Get robot state, images, and joint positions"""
        # Get robot state
        tcp_pose, joint_pos, _, _ = self.robot.get_robot_state()
        
        # Get camera images
        env_img, wrist_img = self._get_camera_frames()
        if env_img is None or wrist_img is None:
            return {
                'tcp_pose': tcp_pose,
                'joint_pos': joint_pos,
                'policy_env_img': None,
                'policy_wrist_img': None,
                'env_img_raw': None,
                'wrist_img_raw': None,
            }
            
        # Process images
        policy_env_img = self.image_processor(torch.from_numpy(env_img.copy()).permute(2, 0, 1))
        policy_wrist_img = self.image_processor(torch.from_numpy(wrist_img.copy()).permute(2, 0, 1))
        
        return {
            'tcp_pose': tcp_pose,
            'joint_pos': joint_pos,
            'policy_env_img': policy_env_img,
            'policy_wrist_img': policy_wrist_img,
            'env_img_raw': env_img.copy(),
            'wrist_img_raw': wrist_img.copy(),
        }
    
    def deploy_action(self, tcp_action, gripper_action):
        self.robot.send_tcp_pose(tcp_action)
        self.gripper.move(gripper_action)
        time.sleep(0.2)

    
    def save_scene_images(self, output_dir, episode_idx):
        """Save scene images to output directory"""
        env_img_rgb, wrist_img_rgb = self._get_camera_frames()
        if env_img_rgb is None or wrist_img_rgb is None:
            return None, None
        Image.fromarray(env_img_rgb).save(f"{output_dir}/env_{episode_idx}.png")
        Image.fromarray(wrist_img_rgb).save(f"{output_dir}/wrist_{episode_idx}.png")
        return env_img_rgb, wrist_img_rgb
    
    def align_with_reference(self, ref_left_img, ref_right_img, raw=False):
        print("=====================================================align_with_reference")
        """Align current scene with reference images"""
        cv2.namedWindow("Left", cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow("Right", cv2.WINDOW_AUTOSIZE)

        while (not input().strip().upper() == 'C'):
            state_data = self.get_robot_state()
        if raw:
            left_img = cv2.cvtColor(state_data['env_img_raw'], cv2.COLOR_RGB2BGR)
            right_img = cv2.cvtColor(state_data['wrist_img_raw'], cv2.COLOR_RGB2BGR)
        else:
            left_img = cv2.cvtColor(
                state_data['policy_env_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8),
                cv2.COLOR_RGB2BGR,
            )
            right_img = cv2.cvtColor(
                state_data['policy_wrist_img'].permute(1, 2, 0).cpu().numpy().astype(np.uint8),
                cv2.COLOR_RGB2BGR,
            )
        cv2.imshow("Left", (np.array(left_img) * 0.5 + np.array(ref_left_img) * 0.5).astype(np.uint8))
        cv2.imshow("Right", (np.array(right_img) * 0.5 + np.array(ref_right_img) * 0.5).astype(np.uint8))
        cv2.waitKey(1)
    
    def align_scene_with_file(self, output_dir, episode_idx):
        """Align current scene with reference images from a given file path"""
        ref_left_img = cv2.imread(f"{output_dir}/env_{episode_idx}.png")
        ref_right_img = cv2.imread(f"{output_dir}/wrist_{episode_idx}.png")
        self.align_with_reference(ref_left_img, ref_right_img, raw=True)
    
    def detach_sigma(self):
        """Detach sigma device and store TCP pose"""
        self.sigma.detach()
        detach_tcp, _, _, _ = self.robot.get_robot_state()
        detach_pos = np.array(detach_tcp[:3])
        detach_rot = R.from_quat(np.array(detach_tcp[3:]), scalar_first=True)
        return detach_pos, detach_rot
    
    def human_teleop_step(self, last_p, last_r):
        """Execute one step of human teleoperation"""
        start_time = time.time()
        
        # Get camera data and robot state
        state_data = self.get_robot_state()
        tcp_pose = state_data['tcp_pose']
        joint_pos = state_data['joint_pos']
        
        # Get teleop controls
        sigma_diff_p, diff_r, width = self.sigma.get_control()
        diff_p = self.robot.init_pose[:3] + sigma_diff_p
        diff_p[2] += self.sigma_z_bias
        diff_r = R.from_quat(self.robot.init_pose[3:7], scalar_first=True) * diff_r

        if self.z_freeze_clutch_threshold is not None:
            clutch = self.controller.get_clutch()
            if clutch < self.z_freeze_clutch_threshold:
                if not self.z_locked:
                    print("lock z", flush=True)
                    self.z_locked = True
                diff_p[2] = last_p[2]
                self.sigma_z_bias = last_p[2] - (self.robot.init_pose[2] + sigma_diff_p[2])
            elif self.z_locked:
                print("release z", flush=True)
                self.z_locked = False

        curr_p_action = diff_p - last_p
        curr_r_action = last_r.inv() * diff_r
        last_p = diff_p
        last_r = diff_r
        
        # Check throttle pedal state (for teleop pausing)
        for event in pygame.event.get():
            if event.type == pygame.QUIT :
                self.keyboard.quit = True
        
        throttle = self.controller.get_throttle()
        if throttle < -0.9:
            if not self.last_throttle:
                self.sigma.detach()
                self.last_throttle = True
            return None, last_p, last_r
        
        if self.last_throttle:
            self.last_throttle = False
            self.sigma.resume()
            last_p, last_r, _ = self.sigma.get_control()
            last_p = last_p + self.robot.init_pose[:3]
            last_r = R.from_quat(self.robot.init_pose[3:7], scalar_first=True) * last_r
            return None, last_p, last_r
        
        # Send command to robot
        self.robot.send_tcp_pose(np.concatenate((diff_p, diff_r.as_quat(scalar_first=True)), 0))
        if self.disable_gripper_cmd:
            gripper_action = self.gripper.get_gripper_state()
        else:
            self.gripper.move_from_sigma(width)
            gripper_action = self.gripper.max_width * width / 1000
        
        # Save demo data for return
        processed_data = {
            'policy_env_img': state_data['policy_env_img'],
            'policy_wrist_img': state_data['policy_wrist_img'],
            'tcp_pose': tcp_pose,
            'joint_pos': joint_pos,
            'action': np.concatenate((curr_p_action, curr_r_action.as_quat(scalar_first=True), [gripper_action])),
            'action_abs': np.concatenate((diff_p, diff_r.as_quat(scalar_first=True), [gripper_action])),
            'action_mode': INTV,
        }
        
        # Sleep to maintain fps
        time.sleep(max(1 / self.fps - (time.time() - start_time), 0))
        
        return processed_data, diff_p, diff_r
    
    def rewind_robot(self, curr_pos, curr_rot, inverse_action):
        """Rewind the robot by applying inverse actions"""

        p_action = inverse_action[:3]
        r_action = inverse_action[3:7]
        gripper_action = inverse_action[7]
        
        # Apply inverse action
        curr_pos = curr_pos - p_action
        curr_rot = curr_rot * R.from_quat(r_action, scalar_first=True).inv()
        
        # Send command
        self.robot.send_tcp_pose(np.concatenate((curr_pos, curr_rot.as_quat(scalar_first=True)), 0))
        self.gripper.move(gripper_action)
        
        return curr_pos, curr_rot


if __name__ == "__main__": # Test the robustness of Sigma teleoperation# t
    # Example usage of RobotEnv
    robot_env = RobotEnv(camera_serial=CAM_SERIAL, img_shape=(3, 224, 224), fps=10)
    robot_env.reset_robot()
    
    last_p = robot_env.robot.init_pose[:3]
    last_r = R.from_quat(robot_env.robot.init_pose[3:7], scalar_first=True)
    
    while True:
        processed_data, last_p, last_r = robot_env.human_teleop_step(last_p, last_r)
        if processed_data is None:
            continue

        # Break condition for demo
        if robot_env.keyboard.quit:
            break
