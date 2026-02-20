"""ROS2 Bridge for OmniNav."""

from __future__ import annotations

from typing import Optional, Dict, Any, TYPE_CHECKING
import time
import warnings
import logging
import math

import numpy as np
from omegaconf import DictConfig

from omninav.interfaces.ros2.adapter import Ros2Adapter
from omninav.interfaces.ros2.components import (
    TopicConfig,
    FrameConfig,
    ClockPublisher,
    TfPublisher,
    OdomPublisher,
    ScanPublisher,
    ImagePublisher,
    CameraInfoPublisher,
    CmdVelSubscriber,
    CmdVelPublisher,
)

if TYPE_CHECKING:
    from omninav.robots.base import RobotBase
    from omninav.core.types import Observation


_PROFILE_DEFAULTS = {
    "all_off": {
        "publish": {
            "clock": False,
            "tf": False,
            "tf_static": False,
            "odom": False,
            "scan": False,
            "rgb_image": False,
            "depth_image": False,
            "camera_info": False,
        },
    },
    "nav2_minimal": {
        "publish": {
            "clock": True,
            "tf": True,
            "tf_static": True,
            "odom": True,
            "scan": True,
            "rgb_image": False,
            "depth_image": False,
            "camera_info": False,
        },
    },
    "nav2_full": {
        "publish": {
            "clock": True,
            "tf": True,
            "tf_static": True,
            "odom": True,
            "scan": True,
            "rgb_image": False,
            "depth_image": False,
            "camera_info": False,
        },
    },
    "rviz_sensors": {
        "publish": {
            "clock": True,
            "tf": True,
            "tf_static": True,
            "odom": True,
            "scan": True,
            "rgb_image": True,
            "depth_image": True,
            "camera_info": True,
        },
    },
}

logger = logging.getLogger(__name__)


class ROS2Bridge:
    """ROS2 bridge for publishing OmniNav state and consuming external commands."""

    def __init__(self, cfg: DictConfig, sim: Any):
        self.cfg = cfg
        self.sim = sim
        self._enabled = bool(cfg.get("enabled", False))
        self._node = None

        self._topics = self._resolve_topics()
        self._frames = self._resolve_frames()
        self._control_source = str(cfg.get("control_source", "python"))
        self._profile = str(cfg.get("profile", "all_off"))
        self._namespace_mode = str(cfg.get("namespace_mode", "single"))
        self._cmd_vel_timeout_sec = float(cfg.get("cmd_vel_timeout_sec", 0.5))
        self._spin_timeout_sec = max(0.0, float(cfg.get("spin_timeout_sec", 0.001)))
        self._warmup_sec = float(cfg.get("warmup_sec", 2.0))
        self._static_tf_republish_steps = int(cfg.get("static_tf_republish_steps", 50))

        self._publish = self._resolve_publish_switches()
        publish_rate_cfg = cfg.get("publish_rate", {})
        self._publish_every_n_steps = {
            "scan": max(1, int(publish_rate_cfg.get("scan_every_n_steps", 1))),
            "rgb_image": max(1, int(publish_rate_cfg.get("rgb_every_n_steps", 1))),
            "depth_image": max(1, int(publish_rate_cfg.get("depth_every_n_steps", 1))),
            "camera_info": max(1, int(publish_rate_cfg.get("camera_info_every_n_steps", 1))),
        }
        self._publish_step = 0

        self._robot: Optional["RobotBase"] = None
        self._clock_pub: Optional[ClockPublisher] = None
        self._tf_pub: Optional[TfPublisher] = None
        self._odom_pub: Optional[OdomPublisher] = None
        self._scan_pub: Optional[ScanPublisher] = None
        self._rgb_pub: Optional[ImagePublisher] = None
        self._depth_pub: Optional[ImagePublisher] = None
        self._rgb_info_pub: Optional[CameraInfoPublisher] = None
        self._depth_info_pub: Optional[CameraInfoPublisher] = None
        self._cmd_sub: Optional[CmdVelSubscriber] = None
        self._cmd_pub: Optional[CmdVelPublisher] = None

        self._cmd_vel: Optional[np.ndarray] = None
        self._last_cmd_vel_ts: Optional[float] = None
        self._static_tf_transforms: list = []
        self._published_static_tf = False
        self._warned_no_camera_data = False

        if self._enabled:
            self._init_ros2()

    def _resolve_topics(self) -> TopicConfig:
        topics = self.cfg.get("topics", {})
        return TopicConfig(
            clock=topics.get("clock", "/clock"),
            tf=topics.get("tf", "/tf"),
            tf_static=topics.get("tf_static", "/tf_static"),
            odom=topics.get("odom", self.cfg.get("odom_topic", "/odom")),
            scan=topics.get("scan", "/scan"),
            rgb_image=topics.get("rgb_image", "/camera/rgb/image_raw"),
            depth_image=topics.get("depth_image", "/camera/depth/image_raw"),
            rgb_camera_info=topics.get("rgb_camera_info", "/camera/rgb/camera_info"),
            depth_camera_info=topics.get("depth_camera_info", "/camera/depth/camera_info"),
            cmd_vel_in=topics.get("cmd_vel_in", self.cfg.get("cmd_vel_topic", "/cmd_vel")),
            cmd_vel_out=topics.get("cmd_vel_out", "/omninav/cmd_vel"),
        )

    def _resolve_frames(self) -> FrameConfig:
        frames = self.cfg.get("frames", {})
        return FrameConfig(
            map=frames.get("map", "map"),
            odom=frames.get("odom", "odom"),
            base_link=frames.get("base_link", "base_link"),
            lidar=frames.get("lidar", frames.get("laser", "laser_frame")),
            rgb_camera=frames.get("rgb_camera", "camera_rgb_frame"),
            depth_camera=frames.get("depth_camera", "camera_depth_frame"),
        )

    def _resolve_publish_switches(self) -> Dict[str, bool]:
        profile_defaults = _PROFILE_DEFAULTS.get(self._profile, _PROFILE_DEFAULTS["all_off"])
        publish_cfg = dict(profile_defaults.get("publish", {}))
        publish_override = self.cfg.get("publish", {})
        for key in ["clock", "tf", "tf_static", "odom", "scan", "rgb_image", "depth_image", "camera_info"]:
            if key in publish_override:
                publish_cfg[key] = bool(publish_override.get(key))
        return publish_cfg

    def _validate_config(self) -> None:
        valid_sources = {"python", "ros2"}
        if self._control_source not in valid_sources:
            raise ValueError(f"Invalid ros2.control_source={self._control_source}; expected one of {sorted(valid_sources)}")

        valid_namespace_modes = {"single", "per_env"}
        if self._namespace_mode not in valid_namespace_modes:
            raise ValueError(
                f"Invalid ros2.namespace_mode={self._namespace_mode}; expected one of {sorted(valid_namespace_modes)}"
            )

    def _build_qos(self, key: str):
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

        default_by_key = {
            "tf_static": {"history": "keep_last", "depth": 1, "reliability": "reliable", "durability": "transient_local"},
            "tf": {"history": "keep_last", "depth": 50, "reliability": "reliable", "durability": "volatile"},
            "clock": {"history": "keep_last", "depth": 10, "reliability": "reliable", "durability": "volatile"},
            "odom": {"history": "keep_last", "depth": 20, "reliability": "reliable", "durability": "volatile"},
            "scan": {"history": "keep_last", "depth": 10, "reliability": "reliable", "durability": "volatile"},
            "rgb_image": {"history": "keep_last", "depth": 5, "reliability": "reliable", "durability": "volatile"},
            "depth_image": {"history": "keep_last", "depth": 5, "reliability": "reliable", "durability": "volatile"},
            "rgb_camera_info": {"history": "keep_last", "depth": 5, "reliability": "reliable", "durability": "volatile"},
            "depth_camera_info": {"history": "keep_last", "depth": 5, "reliability": "reliable", "durability": "volatile"},
            "cmd_vel_in": {"history": "keep_last", "depth": 10, "reliability": "reliable", "durability": "volatile"},
            "cmd_vel_out": {"history": "keep_last", "depth": 10, "reliability": "reliable", "durability": "volatile"},
        }
        defaults = default_by_key.get(key, {"history": "keep_last", "depth": 10, "reliability": "reliable", "durability": "volatile"})
        qos_cfg = self.cfg.get("qos", {}).get(key, {})
        merged = {**defaults, **qos_cfg}

        profile = QoSProfile(depth=int(merged["depth"]))
        profile.history = HistoryPolicy.KEEP_ALL if str(merged["history"]).lower() == "keep_all" else HistoryPolicy.KEEP_LAST
        profile.reliability = (
            ReliabilityPolicy.BEST_EFFORT
            if str(merged["reliability"]).lower() == "best_effort"
            else ReliabilityPolicy.RELIABLE
        )
        profile.durability = (
            DurabilityPolicy.TRANSIENT_LOCAL
            if str(merged["durability"]).lower() == "transient_local"
            else DurabilityPolicy.VOLATILE
        )
        return profile

    def _init_ros2(self) -> None:
        try:
            import rclpy

            self._validate_config()
            if not rclpy.ok():
                rclpy.init()

            node_name = self.cfg.get("node_name", "omninav")
            self._node = rclpy.create_node(node_name)

            # Publishers
            if self._publish.get("clock", False):
                self._clock_pub = ClockPublisher(self._node, self._topics.clock, self._build_qos("clock"))
            if self._publish.get("tf", False) or self._publish.get("tf_static", False):
                self._tf_pub = TfPublisher(
                    self._node,
                    self._topics.tf,
                    self._topics.tf_static,
                    self._build_qos("tf"),
                    self._build_qos("tf_static"),
                )
            if self._publish.get("odom", False):
                self._odom_pub = OdomPublisher(self._node, self._topics.odom, self._build_qos("odom"))
            if self._publish.get("scan", False):
                self._scan_pub = ScanPublisher(self._node, self._topics.scan, self._build_qos("scan"))
            if self._publish.get("rgb_image", False):
                self._rgb_pub = ImagePublisher(self._node, self._topics.rgb_image, self._build_qos("rgb_image"))
            if self._publish.get("depth_image", False):
                self._depth_pub = ImagePublisher(self._node, self._topics.depth_image, self._build_qos("depth_image"))
            if self._publish.get("camera_info", False):
                self._rgb_info_pub = CameraInfoPublisher(
                    self._node, self._topics.rgb_camera_info, self._build_qos("rgb_camera_info")
                )
                self._depth_info_pub = CameraInfoPublisher(
                    self._node, self._topics.depth_camera_info, self._build_qos("depth_camera_info")
                )

            # Subscriptions / optional mirror publisher
            if self._control_source == "ros2":
                self._cmd_sub = CmdVelSubscriber(self._node, self._topics.cmd_vel_in, self._build_qos("cmd_vel_in"), self._cmd_vel_callback)
            self._cmd_pub = CmdVelPublisher(self._node, self._topics.cmd_vel_out, self._build_qos("cmd_vel_out"))

        except ImportError:
            warnings.warn(
                "rclpy not found. ROS2 bridge disabled. Install ROS2 and source setup.bash to enable."
            )
            self._enabled = False

    def setup(self, robot: "RobotBase") -> None:
        self._robot = robot
        if not self._enabled or self._node is None:
            return
        self._publish_static_tf()
        self._warmup_discovery()

    def _get_ros_time(self):
        from builtin_interfaces.msg import Time

        sim_time = float(self.sim.get_sim_time()) if self.sim is not None else 0.0
        sec = int(sim_time)
        nanosec = int((sim_time - sec) * 1e9)
        return Time(sec=sec, nanosec=nanosec)

    def _cmd_vel_callback(self, msg) -> None:
        self._cmd_vel = Ros2Adapter.cmd_vel_from_twist(msg)
        self._last_cmd_vel_ts = time.monotonic()

    def _cmd_vel_fresh(self) -> bool:
        if self._last_cmd_vel_ts is None:
            return False
        return (time.monotonic() - self._last_cmd_vel_ts) <= self._cmd_vel_timeout_sec

    def get_external_cmd_vel(self) -> Optional[np.ndarray]:
        if self._control_source != "ros2":
            return None
        if self._cmd_vel is None:
            return None
        if not self._cmd_vel_fresh():
            return np.zeros(3, dtype=np.float32)
        return self._cmd_vel.copy()

    def spin_once(self) -> None:
        if not self._enabled or self._node is None:
            return

        import rclpy

        rclpy.spin_once(self._node, timeout_sec=self._spin_timeout_sec)

    def _publish_clock(self) -> None:
        if self._clock_pub is None:
            return
        self._clock_pub.publish(self._get_ros_time())

    def _make_odom_transform(self, stamp, robot_state):
        from geometry_msgs.msg import TransformStamped

        pos = np.asarray(robot_state.get("position", np.zeros((1, 3), dtype=np.float32)))
        orient = np.asarray(robot_state.get("orientation", np.array([[1, 0, 0, 0]], dtype=np.float32)))
        if pos.ndim == 2:
            pos = pos[0]
        if orient.ndim == 2:
            orient = orient[0]

        tf = TransformStamped()
        tf.header.stamp = stamp
        tf.header.frame_id = self._frames.odom
        tf.child_frame_id = self._frames.base_link
        tf.transform.translation.x = float(pos[0])
        tf.transform.translation.y = float(pos[1])
        tf.transform.translation.z = float(pos[2])
        tf.transform.rotation.w = float(orient[0])
        tf.transform.rotation.x = float(orient[1])
        tf.transform.rotation.y = float(orient[2])
        tf.transform.rotation.z = float(orient[3])
        return tf

    def _get_sensor_mount(self, sensor_type: str, preferred_names: tuple[str, ...] = ()) -> Optional[tuple[list[float], list[float]]]:
        if self._robot is None or not hasattr(self._robot, "sensors"):
            return None

        named_sensor = None
        for name in preferred_names:
            candidate = self._robot.sensors.get(name, None)
            if candidate is not None:
                named_sensor = candidate
                break

        chosen = named_sensor
        if chosen is None:
            for sensor in self._robot.sensors.values():
                sensor_id = str(getattr(sensor, "SENSOR_TYPE", "")).lower()
                cfg = getattr(sensor, "cfg", {})
                cfg_type = str(cfg.get("type", "")).lower() if hasattr(cfg, "get") else ""
                if sensor_id == sensor_type or cfg_type == sensor_type:
                    chosen = sensor
                    break

        if chosen is None:
            return None

        cfg = getattr(chosen, "cfg", {})
        pos = cfg.get("position", [0.0, 0.0, 0.0])
        ori = cfg.get("orientation", [0.0, 0.0, 0.0])

        if hasattr(pos, "tolist"):
            pos = pos.tolist()
        else:
            pos = list(pos)
        if hasattr(ori, "tolist"):
            ori = ori.tolist()
        else:
            ori = list(ori)
        return pos, ori

    @staticmethod
    def _quat_from_euler_deg(roll_deg: float, pitch_deg: float, yaw_deg: float) -> tuple[float, float, float, float]:
        roll = math.radians(roll_deg)
        pitch = math.radians(pitch_deg)
        yaw = math.radians(yaw_deg)
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        return (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )

    def _build_static_tf_transforms(self):
        """Build the list of static TransformStamped once; reuse for re-publish."""
        from geometry_msgs.msg import TransformStamped

        static_mounts = [
            (self._get_sensor_mount("lidar_2d", preferred_names=("front_lidar", "lidar_2d")), self._frames.lidar),
            (self._get_sensor_mount("camera", preferred_names=("front_camera", "depth_camera", "camera")), self._frames.rgb_camera),
            (self._get_sensor_mount("camera", preferred_names=("front_camera", "depth_camera", "camera")), self._frames.depth_camera),
        ]
        stamp = self._get_ros_time()
        transforms = []
        for mount, child_frame in static_mounts:
            if mount is None:
                continue
            pos, ori = mount
            qw, qx, qy, qz = self._quat_from_euler_deg(float(ori[0]), float(ori[1]), float(ori[2]))
            tf_msg = TransformStamped()
            tf_msg.header.stamp = stamp
            tf_msg.header.frame_id = self._frames.base_link
            tf_msg.child_frame_id = child_frame
            tf_msg.transform.translation.x = float(pos[0])
            tf_msg.transform.translation.y = float(pos[1])
            tf_msg.transform.translation.z = float(pos[2])
            tf_msg.transform.rotation.w = float(qw)
            tf_msg.transform.rotation.x = float(qx)
            tf_msg.transform.rotation.y = float(qy)
            tf_msg.transform.rotation.z = float(qz)
            transforms.append(tf_msg)
        return transforms

    def _publish_static_tf(self) -> None:
        if self._tf_pub is None or not self._publish.get("tf_static", False):
            return

        if not self._static_tf_transforms:
            self._static_tf_transforms = self._build_static_tf_transforms()

        if self._static_tf_transforms:
            self._tf_pub.publish_static_batch(self._static_tf_transforms)
        self._published_static_tf = True

    def _warmup_discovery(self) -> None:
        """Spin for a short period so DDS can complete endpoint discovery.

        On WSL2 the multicast-based discovery is slow; without this warmup,
        RViz may miss the initial static TF and volatile topic data.
        """
        if self._warmup_sec <= 0:
            return

        import rclpy

        logger.info(
            "ROS2Bridge: warming up DDS discovery for %.1fs …", self._warmup_sec,
        )
        deadline = time.monotonic() + self._warmup_sec
        while time.monotonic() < deadline:
            self._publish_clock()
            if self._static_tf_transforms and self._tf_pub is not None:
                self._tf_pub.publish_static_batch(self._static_tf_transforms)
            rclpy.spin_once(self._node, timeout_sec=0.05)
        logger.info("ROS2Bridge: warmup done")

    def publish_observation(self, obs: "Observation") -> None:
        if not self._enabled or self._node is None:
            return

        self._publish_clock()
        robot_state = obs.get("robot_state", None)
        stamp = self._get_ros_time()
        self._publish_step += 1

        if (
            self._publish_step <= self._static_tf_republish_steps
            and self._static_tf_transforms
            and self._tf_pub is not None
        ):
            self._tf_pub.publish_static_batch(self._static_tf_transforms)

        if robot_state is not None and self._odom_pub is not None and self._publish.get("odom", False):
            odom_msg = self._odom_pub.publish(robot_state, stamp, self._frames)
            if self._tf_pub is not None and self._publish.get("tf", False):
                odom_tf = self._make_odom_transform(odom_msg.header.stamp, robot_state)
                self._tf_pub.publish_dynamic(odom_tf)

        if self._scan_pub is not None and self._publish.get("scan", False):
            scan_data = Ros2Adapter.pick_scan_data(obs.get("sensors", {}))
            if scan_data is not None and (self._publish_step % self._publish_every_n_steps["scan"] == 0):
                scan_data = Ros2Adapter.enrich_scan_data(scan_data)
                self._scan_pub.publish(scan_data, stamp, self._frames.lidar)

        cam_data = Ros2Adapter.pick_camera_data(obs.get("sensors", {}))
        if cam_data is not None:
            self._warned_no_camera_data = False
            if (
                self._rgb_pub is not None
                and self._publish.get("rgb_image", False)
                and "rgb" in cam_data
                and (self._publish_step % self._publish_every_n_steps["rgb_image"] == 0)
            ):
                rgb = Ros2Adapter.normalize_rgb_image(cam_data["rgb"])
                self._rgb_pub.publish_rgb8(rgb, stamp, self._frames.rgb_camera)
                if (
                    self._rgb_info_pub is not None
                    and self._publish.get("camera_info", False)
                    and (self._publish_step % self._publish_every_n_steps["camera_info"] == 0)
                ):
                    fov = float(cam_data.get("fov", 60.0))
                    self._rgb_info_pub.publish(rgb.shape[0], rgb.shape[1], stamp, self._frames.rgb_camera, fov)
            if (
                self._depth_pub is not None
                and self._publish.get("depth_image", False)
                and "depth" in cam_data
                and (self._publish_step % self._publish_every_n_steps["depth_image"] == 0)
            ):
                depth = Ros2Adapter.normalize_depth_image(cam_data["depth"])
                self._depth_pub.publish_depth32f(depth, stamp, self._frames.depth_camera)
                if (
                    self._depth_info_pub is not None
                    and self._publish.get("camera_info", False)
                    and (self._publish_step % self._publish_every_n_steps["camera_info"] == 0)
                ):
                    fov = float(cam_data.get("fov", 60.0))
                    self._depth_info_pub.publish(depth.shape[0], depth.shape[1], stamp, self._frames.depth_camera, fov)
        elif (self._publish.get("rgb_image", False) or self._publish.get("depth_image", False)) and not self._warned_no_camera_data:
            self._warned_no_camera_data = True
            logger.warning("ROS2Bridge camera publish enabled, but no camera data found. sensor keys=%s", list(obs.get("sensors", {}).keys()))

    def publish_cmd_vel(self, cmd_vel: np.ndarray) -> None:
        if not self._enabled or self._cmd_pub is None:
            return
        self._cmd_pub.publish(Ros2Adapter.first_batch(cmd_vel))

    def shutdown(self) -> None:
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
        try:
            import rclpy

            if rclpy.ok():
                rclpy.shutdown()
        except ImportError:
            pass

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def control_source(self) -> str:
        return self._control_source
