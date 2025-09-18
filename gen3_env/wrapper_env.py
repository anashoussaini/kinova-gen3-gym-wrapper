from __future__ import annotations
import time, math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple, List

import numpy as np
from PIL import Image as PILImage

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
from rclpy.action import ActionClient

from sensor_msgs.msg import JointState, Image
from geometry_msgs.msg import Twist
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import GripperCommand
from cv_bridge import CvBridge


# ----------------------------- Config -----------------------------

@dataclass
class Gen3Config:
    # YOU keep controllers active; this wrapper only reads/publishes
    control_mode: str = "joint_trajectory"  # "twist" | "joint_trajectory"
    rate_hz: float = 15.0

    # Topics
    joint_states_topic: str = "/joint_states"
    external_rgb_topic: Optional[str] = "/external_cam/camera/color/image_raw"
    wrist_rgb_topic:    Optional[str] = "/wrist_cam/camera/color/image_raw"
    twist_cmd_topic: str = "/twist_controller/commands"
    joint_traj_topic: str = "/joint_trajectory_controller/joint_trajectory"

    # Arm joint names (7-DoF Gen3)
    joint_names: Sequence[str] = field(default_factory=lambda: (
        "joint_1","joint_2","joint_3","joint_4","joint_5","joint_6","joint_7"
    ))

    # Robotiq gripper (Action server)
    robotiq_action_name: Optional[str] = "/robotiq_gripper_controller/gripper_cmd"
    gripper_min_pos: float = 0.0      # 0.0=open
    gripper_max_pos: float = 0.8      # ~0.8=close (typical Robotiq)
    gripper_default_effort: float = 100.0

    # Optional homing via joint-trajectory (no services)
    home_on_reset: bool = True
    home_joint_positions: Optional[Sequence[float]] = None  # 7 values, radians
    home_time_sec: float = 4.0

    # Limits / smoothing
    lin_vel_limit: float = 0.08  # m/s
    ang_vel_limit: float = 0.5   # rad/s
    joint_step_limit: float = math.radians(10.0)  # per-step arm move cap
    action_smoothing_tau: float = 0.0  # seconds; EMA (0 disables)

    # Data readiness
    wait_topics_timeout: float = 10.0  # secs to wait at reset()


# --------------------------- Environment --------------------------

class Gen3Env:
    """
    Minimal ROS2 wrapper for Kinova Gen3:
    - observe(): RGB external/wrist + 7-DoF joint state (ordered by cfg.joint_names)
    - step(action[, seconds]):
        joint_trajectory -> len=7 (arm) or len=8 (arm+gripper)
        twist            -> len=6 (twist) or len=7 (twist+gripper)
      Gripper (last value) is sent to Robotiq action if configured.
    """
    def __init__(self, cfg: Gen3Config):
        self.cfg = cfg
        self._closed = False

        # init ROS + build I/O
        if not rclpy.ok():
            rclpy.init()
        self._build_ros_io()

        # Timing / smoothing
        self._dt = 1.0 / max(self.cfg.rate_hz, 1e-6)
        self._ema_alpha = 0.0 if self.cfg.action_smoothing_tau <= 0 else (
            1.0 - math.exp(-self._dt / self.cfg.action_smoothing_tau)
        )
        self._last_twist = np.zeros(6, dtype=np.float32)
        self._last_grip_cmd = self.cfg.gripper_min_pos

    # ---------- ROS I/O builders & recovery ----------

    def _build_ros_io(self):
        # Node
        self.node = Node("gen3_gym_env")

        # QoS for images
        self._qos_img = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=2,
        )

        # Bridges
        self.bridge = CvBridge()

        # State
        self._joint_state: Optional[JointState] = None
        self._arm_idx: Optional[List[int]] = None
        self._external_img: Optional[np.ndarray] = None
        self._wrist_img: Optional[np.ndarray] = None

        # Subs
        self.node.create_subscription(JointState, self.cfg.joint_states_topic, self._on_joint_state, 10)
        if self.cfg.external_rgb_topic:
            self.node.create_subscription(Image, self.cfg.external_rgb_topic, self._on_external_img, self._qos_img)
        if self.cfg.wrist_rgb_topic:
            self.node.create_subscription(Image, self.cfg.wrist_rgb_topic, self._on_wrist_img, self._qos_img)

        # Pubs
        self.twist_pub  = self.node.create_publisher(Twist,           self.cfg.twist_cmd_topic,  10)
        self.jtraj_pub  = self.node.create_publisher(JointTrajectory, self.cfg.joint_traj_topic, 10)

        # Gripper action
        self.grip_client: Optional[ActionClient] = None
        if self.cfg.robotiq_action_name:
            self.grip_client = ActionClient(self.node, GripperCommand, self.cfg.robotiq_action_name)

    def _rebuild_ros_io(self):
        # Hard rebuild: destroy old node (ignore errors), re-init rclpy if needed, then rebuild everything
        try:
            self.node.destroy_node()
        except Exception:
            pass
        if not rclpy.ok():
            rclpy.init()
        self._build_ros_io()

    def _ensure_alive(self):
        """Recover from 'InvalidHandle' scenarios (e.g., after shutdown or notebook reload)."""
        if self._closed:
            # If user called close(), don't auto-revive (explicit is better)
            raise RuntimeError("Gen3Env is closed. Create a new env or call reopen().")
        if not rclpy.ok():
            rclpy.init()
        # Probe a handle quickly — if invalid, rebuild
        try:
            with self.jtraj_pub.handle:
                pass
        except Exception:
            self._rebuild_ros_io()

    def reopen(self):
        """Explicitly rebuild the ROS node/pubs/subs if you called close() earlier."""
        self._closed = False
        self._rebuild_ros_io()

    # --------------------------- Public API ---------------------------

    def reset(self) -> Dict[str, np.ndarray]:
        """Optional home (joint trajectory) + wait for data; returns first obs."""
        self._ensure_alive()

        if self.cfg.home_on_reset and self.cfg.home_joint_positions is not None:
            self._publish_joint_trajectory_safe(list(self.cfg.home_joint_positions), self.cfg.home_time_sec)

        self._wait_for_data(
            need_joint=True,
            need_ext=self.cfg.external_rgb_topic is not None,
            need_wrist=self.cfg.wrist_rgb_topic is not None,
            timeout=self.cfg.wait_topics_timeout,
        )
        return self.observe()

    def observe(self) -> Dict[str, np.ndarray]:
        obs: Dict[str, np.ndarray] = {}
        if self._external_img is not None:
            obs["rgb_external"] = self._external_img
        if self._wrist_img is not None:
            obs["rgb_wrist"] = self._wrist_img

        if self._joint_state is not None:
            q, dq, tau = self._arm_from_joint_state(self._joint_state)
            obs["joint_position"] = q
            if dq is not None:  obs["joint_velocity"] = dq
            if tau is not None: obs["joint_effort"]   = tau

        obs["gripper_commanded"] = np.array([self._last_grip_cmd], dtype=np.float32)
        return obs

    def step(self, action: Sequence[float], seconds: Optional[float] = None) -> Dict[str, np.ndarray]:
        """
        joint_trajectory:
          - len=7  -> arm absolute targets
          - len>=8 -> first 7 arm, last is gripper in [min,max]
        twist:
          - len=6  -> twist
          - len>=7 -> first 6 twist, last is gripper in [min,max]
        """
        self._ensure_alive()

        a = np.asarray(action, np.float32).reshape(-1)
        grip_val = None

        if self.cfg.control_mode == "joint_trajectory":
            n = len(self.cfg.joint_names)
            if a.size < n:
                raise ValueError(f"Joint mode expects at least {n} values (got {a.size}).")
            arm = a[:n]
            if a.size >= n + 1:
                grip_val = float(a[n])
            self._step_joint_traj_safe(arm, seconds)
        elif self.cfg.control_mode == "twist":
            if a.size < 6:
                raise ValueError(f"Twist mode expects at least 6 values (got {a.size}).")
            twist_cmd = a[:6]
            if a.size >= 7:
                grip_val = float(a[6])
            self._step_twist_safe(twist_cmd, seconds)
        else:
            raise ValueError(f"Unsupported control_mode: {self.cfg.control_mode}")

        if grip_val is not None:
            self.set_gripper(grip_val)

        return self.observe()

    # ----- Gripper helpers -----

    def open_gripper(self):
        self.set_gripper(self.cfg.gripper_min_pos)

    def close_gripper(self):
        self.set_gripper(self.cfg.gripper_max_pos)

    def set_gripper(self, position: float, max_effort: Optional[float] = None):
        """Send a Robotiq GripperCommand action goal. Non-blocking; auto-recovers if needed."""
        self._ensure_alive()

        self._last_grip_cmd = float(np.clip(position, self.cfg.gripper_min_pos, self.cfg.gripper_max_pos))
        if self.grip_client is None:
            self.node.get_logger().warn("Robotiq action client not configured; skipping set_gripper().")
            return

        def _send():
            goal = GripperCommand.Goal()
            goal.command.position = float(self._last_grip_cmd)
            goal.command.max_effort = float(self.cfg.gripper_default_effort if max_effort is None else max_effort)
            self.grip_client.send_goal_async(goal)

        # Try once; if server not up or handle invalid, rebuild and try again once
        try:
            if not self.grip_client.wait_for_server(timeout_sec=0.5):
                self.node.get_logger().warn("Robotiq gripper action server not available.")
                return
            _send()
        except Exception:
            self._rebuild_ros_io()
            try:
                if self.grip_client and self.grip_client.wait_for_server(timeout_sec=0.2):
                    _send()
                else:
                    self.node.get_logger().warn("Gripper server still unavailable after recovery.")
            except Exception as e:
                self.node.get_logger().warn(f"Failed to send gripper goal after recovery: {e}")

    def close(self):
        # best-effort zero twist
        try:
            self.twist_pub.publish(Twist())
        except Exception:
            pass
        try:
            self.node.destroy_node()
        finally:
            self._closed = True
            # DO NOT call rclpy.shutdown() here: user might have other nodes alive

    # ------------------------ Internals --------------------------

    def _on_joint_state(self, msg: JointState):
        self._joint_state = msg
        if self._arm_idx is None and msg.name:
            name_to_idx = {n: i for i, n in enumerate(msg.name)}
            idxs = []
            for jn in self.cfg.joint_names:
                if jn in name_to_idx:
                    idxs.append(name_to_idx[jn])
                else:
                    self.node.get_logger().warn(f"Joint '{jn}' not in JointState; using first 7 fallback.")
                    self._arm_idx = None
                    return
            self._arm_idx = idxs

    def _on_external_img(self, msg: Image):
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self._external_img = _bgr2rgb(img_bgr)
        except Exception as e:
            self.node.get_logger().error(f"external cv_bridge error: {e}")

    def _on_wrist_img(self, msg: Image):
        try:
            img_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self._wrist_img = _bgr2rgb(img_bgr)
        except Exception as e:
            self.node.get_logger().error(f"wrist cv_bridge error: {e}")

    def _wait_for_data(self, need_joint: bool, need_ext: bool, need_wrist: bool, timeout: float):
        t0 = time.time()
        while time.time() - t0 < timeout:
            ok = True
            if need_joint and self._joint_state is None: ok = False
            if need_ext  and self._external_img is None: ok = False
            if need_wrist and self._wrist_img is None:   ok = False
            if ok: return
            rclpy.spin_once(self.node, timeout_sec=0.05)
            time.sleep(0.05)
        self.node.get_logger().warn("Timed out waiting for initial data (continuing).")

    # ---- control paths (safe, auto-recover) ----

    def _step_twist_safe(self, action6: Sequence[float], seconds: Optional[float]):
        try:
            self._step_twist(action6, seconds)
        except Exception:
            self._rebuild_ros_io()
            self._step_twist(action6, seconds)

    def _step_joint_traj_safe(self, arm7: Sequence[float], seconds: Optional[float]):
        try:
            self._step_joint_traj(arm7, seconds)
        except Exception:
            self._rebuild_ros_io()
            self._step_joint_traj(arm7, seconds)

    def _step_twist(self, action6: Sequence[float], seconds: Optional[float]):
        a = np.asarray(action6, np.float32).reshape(-1)
        if a.size < 6:
            raise ValueError("Twist expects 6 values [vx,vy,vz,wx,wy,wz].")
        v_lin = np.clip(a[:3], -self.cfg.lin_vel_limit, self.cfg.lin_vel_limit)
        v_ang = np.clip(a[3:6], -self.cfg.ang_vel_limit, self.cfg.ang_vel_limit)
        act = np.concatenate([v_lin, v_ang], axis=0)
        if self._ema_alpha > 0:
            self._last_twist = (1 - self._ema_alpha) * self._last_twist + self._ema_alpha * act
            act = self._last_twist

        duration = self._dt if seconds is None else max(0.0, float(seconds))
        end = time.time() + duration
        msg = Twist()
        while time.time() < end:
            msg.linear.x, msg.linear.y, msg.linear.z = float(act[0]), float(act[1]), float(act[2])
            msg.angular.x, msg.angular.y, msg.angular.z = float(act[3]), float(act[4]), float(act[5])
            self._publish_twist_safe(msg)
            rclpy.spin_once(self.node, timeout_sec=0.0)
            time.sleep(self._dt)

    def _publish_twist_safe(self, msg: Twist):
        try:
            self.twist_pub.publish(msg)
        except Exception:
            self._rebuild_ros_io()
            self.twist_pub.publish(msg)

    def _step_joint_traj(self, arm7: Sequence[float], seconds: Optional[float]):
        a = np.asarray(arm7, np.float32).reshape(-1)
        n = len(self.cfg.joint_names)
        if a.size != n:
            raise ValueError(f"Joint trajectory expects {n} values (got {a.size}).")
        target = a.copy()
        cur = None
        if self._joint_state is not None:
            q, _, _ = self._arm_from_joint_state(self._joint_state)
            cur = q
        if cur is not None:
            delta = np.clip(target - cur, -self.cfg.joint_step_limit, self.cfg.joint_step_limit)
            target = (cur + delta).astype(np.float32)

        move_time = max(0.5, float(seconds) if seconds is not None else 1.5)
        self._publish_joint_trajectory_safe(target.tolist(), duration=move_time)

    def _publish_joint_trajectory_safe(self, positions: List[float], duration: float):
        msg = JointTrajectory()
        msg.joint_names = list(self.cfg.joint_names)
        pt = JointTrajectoryPoint()
        pt.positions = [float(x) for x in positions]
        pt.time_from_start = Duration(seconds=duration).to_msg()
        msg.points = [pt]
        try:
            self.jtraj_pub.publish(msg)
        except Exception:
            self._rebuild_ros_io()
            self.jtraj_pub.publish(msg)

    # ---------------------- helpers ----------------------

    def _arm_from_joint_state(self, js: JointState):
        """
        Returns (q, dq, tau) for the 7 arm joints, in cfg.joint_names order.
        Falls back to first 7 values if names don't match.
        """
        n = len(self.cfg.joint_names)
        if self._arm_idx is not None and js.position and len(js.position) > max(self._arm_idx):
            q = np.array([js.position[i] for i in self._arm_idx], dtype=np.float32)
            dq = np.array([js.velocity[i] for i in self._arm_idx], dtype=np.float32) if js.velocity else None
            tau = np.array([js.effort[i]   for i in self._arm_idx], dtype=np.float32) if js.effort   else None
            return q, dq, tau
        # fallback: take first 7
        q = np.array(js.position[:n], dtype=np.float32) if js.position else np.zeros((n,), np.float32)
        dq = np.array(js.velocity[:n], dtype=np.float32) if js.velocity else None
        tau = np.array(js.effort[:n],   dtype=np.float32) if js.effort   else None
        return q, dq, tau

    # --------- optional 224 utilities ---------

    @staticmethod
    def to_224_center_crop(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        s = min(h, w)
        y0, x0 = (h - s)//2, (w - s)//2
        sq = img[y0:y0+s, x0:x0+s]
        pil = PILImage.fromarray(sq if sq.dtype == np.uint8 else np.clip(sq*255,0,255).astype(np.uint8))
        return np.array(pil.resize((224, 224), PILImage.BILINEAR))

    @staticmethod
    def to_224_square_pad(img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        h, w = img.shape[:2]
        s = max(h, w)
        t, b = (s-h)//2, s-h - (s-h)//2
        l, r = (s-w)//2, s-w - (s-w)//2
        padded = np.pad(img, ((t,b),(l,r),(0,0)), mode="constant", constant_values=0)
        pil = PILImage.fromarray(padded if padded.dtype == np.uint8 else np.clip(padded*255,0,255).astype(np.uint8))
        return np.array(pil.resize((224, 224), PILImage.BILINEAR))


# --------------------------- Factory ---------------------------

def make(env_id: str, control_mode: str, **kwargs) -> Gen3Env:
    if env_id.lower() != "gen3":
        raise ValueError(f"Unknown env_id '{env_id}'. Only 'gen3' is supported.")
    cfg = Gen3Config(control_mode=control_mode, **kwargs)
    return Gen3Env(cfg)


# --------------------------- Utils ---------------------------

def _bgr2rgb(bgr_img) -> np.ndarray:
    import cv2
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
