import time
import numpy as np

import sys
import os

# sys.path.insert(0, os.path.join(os.path.dirname(__file__), "flexiv_rdk", "lib_py"))

# Import Flexiv RDK Python library
import flexivrdk
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from .get_ip import get_local_ip
except ImportError:
    from get_ip import get_local_ip
from my_device.macros import ROBOT_SN, HOME_JOINT_DEG
from utils.transformation import rotation_transform, xyz_rot_to_mat  # type: ignore


def _pose7_to_mat(pose7: np.ndarray) -> np.ndarray:
    """Convert pose [x,y,z,rw,rx,ry,rz] to 4x4 matrix."""
    pose7 = np.asarray(pose7, dtype=np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = pose7[:3]
    # flexiv uses (rw, rx, ry, rz)
    quat = pose7[3:7]
    # convert quaternion to rotation matrix via rotation_transform
    R = rotation_transform(quat[None, :], 'quaternion', 'matrix').squeeze(0)
    T[:3, :3] = R
    return T


def _mat_to_pose7(T: np.ndarray) -> np.ndarray:
    """Convert 4x4 matrix to pose [x,y,z,rw,rx,ry,rz]."""
    T = np.asarray(T, dtype=np.float32)
    xyz = T[:3, 3]
    quat = rotation_transform(T[:3, :3][None, ...], 'matrix', 'quaternion').squeeze(0)
    return np.concatenate([xyz, quat], axis=0)


def compose_relative_delta(curr_pose7: np.ndarray, delta_xyz_rot6d: np.ndarray) -> np.ndarray:
    """Compose current tcp pose with a relative delta (xyz+rot6d) and return target pose7.

    - curr_pose7: [x,y,z,rw,rx,ry,rz]
    - delta_xyz_rot6d: [dx,dy,dz,r6d...]
    """
    T_curr = _pose7_to_mat(curr_pose7)
    # xyz_rot_to_mat expects concatenated xyz + rotation representation
    T_delta = xyz_rot_to_mat(delta_xyz_rot6d.astype(np.float32), rotation_rep='rotation_6d')
    T_target = T_curr @ T_delta
    return _mat_to_pose7(T_target)


def compose_global_delta(curr_pose7: np.ndarray, delta_xyz_rot6d: np.ndarray) -> np.ndarray:
    """Compose current tcp pose with a global delta (xyz+rot6d) and return target pose7.

    - curr_pose7: [x,y,z,rw,rx,ry,rz]
    - delta_xyz_rot6d: [dx,dy,dz,r6d...]
    """
    T_curr = _pose7_to_mat(curr_pose7)
    # xyz_rot_to_mat expects concatenated xyz + rotation representation
    T_delta = xyz_rot_to_mat(delta_xyz_rot6d.astype(np.float32), rotation_rep='rotation_6d')
    # Keep translation unchanged; only apply rotation
    T_delta[:3, 3] = 0.0
    T_target = T_delta @ T_curr
    T_target[:3, 3] = T_curr[:3, 3]
    return _mat_to_pose7(T_target)




class ModeMap:
    idle = "IDLE"
    cart_impedance_online = "NRT_CARTESIAN_MOTION_FORCE"
    joint = "NRT_JOINT_POSITION"


class FlexivRobot:
    """
    Flexiv Robot Control Class.
    """
    logger_name = "FlexivRobot"

    def __init__(
        self,
        robot_ip_address=None,
        pc_ip_address=None,
        default_pose=[0.45,0,0.35,0.707,0,0.707,0],
        home=True,
        connect_retries=3,
        connect_retry_interval=2.0,
        robot_sn=None,
    ):
        if pc_ip_address is None and robot_ip_address is not None:
            pc_ip_address = get_local_ip(robot_ip_address)
        
        self.robot_states = flexivrdk.RobotStates()
        # self.log = flexivrdk.Log()
        self.mode = flexivrdk.Mode
        self.robot = self._connect_robot(
            robot_sn=robot_sn or ROBOT_SN,
            connect_retries=connect_retries,
            connect_retry_interval=connect_retry_interval,
        )
        self.default_pose = default_pose
        self.home_pose = self.default_pose
        self.home_joint_pos = np.deg2rad(HOME_JOINT_DEG).tolist()
        self.init_robot(home)
        self.init_pose = self.get_tcp_pose()

    @staticmethod
    def _connect_robot(robot_sn, connect_retries=3, connect_retry_interval=2.0):
        last_exc = None
        for attempt in range(1, int(connect_retries) + 1):
            try:
                return flexivrdk.Robot(robot_sn)
            except RuntimeError as exc:
                last_exc = exc
                if attempt >= int(connect_retries):
                    break
                print(
                    f"[FlexivRobot] Robot connection failed ({attempt}/{connect_retries}): {exc}. "
                    f"Retrying in {connect_retry_interval}s..."
                )
                time.sleep(float(connect_retry_interval))
        raise last_exc

    def init_robot(self, home = True):
        # log = self.log
        mode = self.mode
        robot = self.robot

        # Clear fault on robot server if any
        if robot.fault():
            # log.warn("Fault occurred on robot server, trying to clear ...")
            print("Fault occurred on robot server, trying to clear ...")
            # Try to clear the fault
            robot.ClearFault()
            time.sleep(2)
            # Check again
            if robot.fault():
                # log.error("Fault cannot be cleared, exiting ...")
                print("Fault cannot be cleared, exiting ...")
                return
            # log.info("Fault on robot server is cleared")
            print("Fault on robot server is cleared")

        # Enable the robot, make sure the E-stop is released before enabling
        # log.info("Enabling robot ...")
        print("Enabling robot ...")
        # robot.enable()
        robot.Enable()

        # Wait for the robot to become operational
        while not robot.operational():
            time.sleep(1)

        # log.info("Robot is now operational")
        print("Robot is now operational")

        # Move robot to home pose
        # log.info("Moving to home pose")
        print("Moving to home pose")
        # self.send_joint_pose(self.home_joint_pos)
        # time.sleep(4)
        if home:
            #self.send_tcp_pose(self.home_pose)
            self.send_joint_pose(self.home_joint_pos)
            time.sleep(4)

        robot.SwitchMode(mode.NRT_PRIMITIVE_EXECUTION)
        # Zero Force-torque Sensor
        # =========================================================================================
        # IMPORTANT: must zero force/torque sensor offset for accurate force/torque measurement
        robot.ExecutePrimitive("ZeroFTSensor", {}, block_until_started = False)

        # WARNING: during the process, the robot must not contact anything, otherwise the result
        # will be inaccurate and affect following operations
        # log.warn(
        #     "Zeroing force/torque sensors, make sure nothing is in contact with the robot"
        # )
        print("Zeroing force/torque sensors, make sure nothing is in contact with the robot")
        # Wait for primitive completion
        while robot.busy():
            time.sleep(0.1)
        # log.info("Sensor zeroing complete")
        print("Sensor zeroing complete")
        # Return controller to idle so downstream code starts from non-busy state
        robot.SwitchMode(mode.IDLE)

    def enable(self, max_time=10):
        """Enable robot after emergency button is released."""
        self.robot.Enable()
        tic = time.time()
        while not self.is_operational():
            if time.time() - tic > max_time:
                return "Robot enable failed"
            time.sleep(0.01)
        return

    def _get_robot_status(self):
        self.robot_states = self.robot.states()
        return self.robot_states

    def mode_mapper(self, mode):
        assert mode in ModeMap.__dict__.keys(), "unknown mode name: %s" % mode
        return getattr(self.mode, getattr(ModeMap, mode))

    def get_control_mode(self):
        return self.robot.mode()

    def set_control_mode(self, mode):
        control_mode = self.mode_mapper(mode)
        self.robot.SwitchMode(control_mode)

    def switch_mode(self, mode, sleep_time=0.01):
        """switch to different control modes.

        Args:
            mode: 'idle', 'cart_impedance_online'
            sleep_time: sleep time to control mode switch time

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        if self.get_control_mode() == self.mode_mapper(mode):
            return

        while self.get_control_mode() != self.mode_mapper("idle"):
            self.set_control_mode("idle")
            time.sleep(sleep_time)
        while self.get_control_mode() != self.mode_mapper(mode):
            self.set_control_mode(mode)
            time.sleep(sleep_time)

        print("[Robot] Set mode: {}".format(str(self.get_control_mode())))

    def clear_fault(self):
        self.robot.ClearFault()

    def is_fault(self):
        """Check if robot is in FAULT state."""
        # return self.robot.isFault()
        return self.robot.fault()

    def is_stopped(self):
        """Check if robot is stopped."""
        return self.robot.stopped()

    def is_connected(self):
        """return if connected.

        Returns: True/False
        """
        return self.robot.connected()

    def is_operational(self):
        """Check if robot is operational."""
        return self.robot.operational()

    def get_tcp_pose(self):
        """get current robot's tool pose in world frame.

        Returns:
            7-dim list consisting of (x,y,z,rw,rx,ry,rz)

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self._get_robot_status().tcp_pose)

    def get_tcp_vel(self):
        """get current robot's tool velocity in world frame.

        Returns:
            7-dim list consisting of (vx,vy,vz,vrw,vrx,vry,vrz)

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self._get_robot_status().tcp_vel)

    def get_joint_pos(self):
        """get current joint value.

        Returns:
            7-dim numpy array of 7 joint position

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self._get_robot_status().q)

    def get_joint_vel(self):
        """get current joint velocity.

        Returns:
            7-dim numpy array of 7 joint velocity

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        return np.array(self._get_robot_status().dq)

    def stop(self):
        """Stop current motion and switch mode to idle."""
        self.robot.Stop()
        while self.get_control_mode() != self.mode_mapper("idle"):
            time.sleep(0.005)

    def set_max_contact_wrench(self, max_wrench):
        self.switch_mode('cart_impedance_online')
        self.robot.setMaxContactWrench(max_wrench)

    def send_impedance_online_pose(self, tcp):
        """make robot move towards target pose in impedance control mode,
        combining with sleep time makes robot move smmothly.

        Args:
            tcp: 7-dim list or numpy array, target pose (x,y,z,rw,rx,ry,rz) in world frame
            wrench: 6-dim list or numpy array, max moving force (fx,fy,fz,wx,wy,wz)

        Raises:
            RuntimeError: error occurred when mode is None.
        """
        self.switch_mode('cart_impedance_online')
        self.robot.SendCartesianMotionForce(np.array(tcp), [0] * 6, max_linear_vel=0.1) # 0.1: maximum velocity

    def send_tcp_pose(self, tcp):
        """
        Send tcp pose.
        """
        self.send_impedance_online_pose(tcp)

    def send_joint_pose(self, q):
        """
        Send joint pose.
        """
        self.switch_mode("joint")
        q = np.asarray(q, dtype=np.float64).tolist()
        dof = len(q)
        target_vel = [0.0] * dof
        max_vel = [1.0] * dof
        max_acc = [1.0] * dof
        self.robot.SendJointPosition(q, target_vel, max_vel, max_acc)


    def get_robot_state(self):
        raw = self._get_robot_status()
        tcpPose = raw.tcp_pose
        tcpVel = raw.tcp_vel
        jointPose = raw.q
        jointVel = raw.dq
        return tcpPose, jointPose, tcpVel, jointVel
    
    def busy(self):
        return self.robot.busy()
    
class FlexivGripper:
    def __init__(
        self,
        r: FlexivRobot,
        *,
        move_retries=3,
        retry_interval=0.1,
        min_command_interval=0.2,
        width_tolerance=0.001,
    ) -> None:
        self.gripper_state = flexivrdk.GripperStates()
        self.gripper = flexivrdk.Gripper(r.robot)
        self.gripper.Enable("Robotiq-2F-85")
        self.gripper_state = self.gripper.states()
        print(self.gripper_state.width)
        self.max_width = 0.085
        self.move_retries = int(move_retries)
        self.retry_interval = float(retry_interval)
        self.min_command_interval = float(min_command_interval)
        self.width_tolerance = float(width_tolerance)
        self._last_command_width = None
        self._last_command_time = 0.0

    def move(self, width):
        # self.gripper.move(self.max_width * width / 1000, 0.1, 20)
        width = float(np.clip(width, 0, self.max_width))
        self._send_move(width, 0.1, 25)

    def move_from_sigma(self, width):
        width = float(np.clip(self.max_width * width / 1000, 0, self.max_width))
        self._send_move(width, 0.1, 25)

    def _send_move(self, width, velocity, force):
        now = time.monotonic()
        if self._last_command_width is not None:
            width_delta = abs(width - self._last_command_width)
            if width_delta < self.width_tolerance:
                return
            if now - self._last_command_time < self.min_command_interval:
                return

        last_exc = None
        for attempt in range(1, self.move_retries + 1):
            try:
                self.gripper.Move(float(width), float(velocity), float(force))
                self._last_command_width = float(width)
                self._last_command_time = time.monotonic()
                return
            except RuntimeError as exc:
                last_exc = exc
                if attempt >= self.move_retries:
                    break
                print(
                    f"[FlexivGripper] Move failed ({attempt}/{self.move_retries}): {exc}. "
                    f"Retrying in {self.retry_interval}s..."
                )
                time.sleep(self.retry_interval)
        raise last_exc

    def get_gripper_state(self):
        self.gripper_state = self.gripper.states()
        return self.gripper_state.width  
    
if __name__ == "__main__":
    robot = FlexivRobot()
    gripper = FlexivGripper(robot)

    # 获取初始位姿，作为运动中心点
    center_pose = robot.get_tcp_pose().copy()
    print("[INFO] Start repetitive motion around:", center_pose)

    # 设置运动幅度和速度
    delta = 0.05  # 每次偏移 5cm
    sleep_time = 0.01

    # 定义两个往返目标点（在 X 方向前后移动）
    pose_forward = center_pose.copy()
    pose_forward[2] += delta
    pose_backward = center_pose.copy()
    pose_backward[2] -= delta

    try:
        while True:
            # 向前
            print("[INFO] Moving forward...")
            robot.send_tcp_pose(pose_forward)
            gripper.move(0.085)  # 张开夹爪
            time.sleep(2)
            # 打印当前位置，用于观察
            tcp_pose, joint_pos, _, _ = robot.get_robot_state()
            print("[Current TCP Pose]:", np.round(tcp_pose, 4))
            # 向后
            print("[INFO] Moving backward...")
            robot.send_tcp_pose(pose_backward)
            gripper.move(0)  # 夹紧夹爪
            time.sleep(2)

            # 打印当前位置，用于观察
            tcp_pose, joint_pos, _, _ = robot.get_robot_state()
            print("[Current TCP Pose]:", np.round(tcp_pose, 4))

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user (Ctrl+C). Moving to safe position...")
        robot.send_tcp_pose(center_pose)
        time.sleep(2)
        robot.stop()
        print("[INFO] Robot stopped safely.")
