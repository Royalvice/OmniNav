"""ROS2 Bridge for OmniNav."""

from __future__ import annotations

from typing import Optional, Dict, Any, TYPE_CHECKING
import time
import warnings

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
    CmdVelSubscriber,
    CmdVelPublisher,
)

if TYPE_CHECKING:
    from omninav.robots.base import RobotBase
    from omninav.core.types import Observation


_PROFILE_DEFAULTS = {
    "all_off": {
        "publish": {"clock": False, "tf": False, "tf_static": False, "odom": False, "scan": False},
    },
    "nav2_minimal": {
        "publish": {"clock": True, "tf": True, "tf_static": True, "odom": True, "scan": True},
    },
    "nav2_full": {
        "publish": {"clock": True, "tf": True, "tf_static": True, "odom": True, "scan": True},
    },
}


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

        self._publish = self._resolve_publish_switches()

        self._robot: Optional["RobotBase"] = None
        self._clock_pub: Optional[ClockPublisher] = None
        self._tf_pub: Optional[TfPublisher] = None
        self._odom_pub: Optional[OdomPublisher] = None
        self._scan_pub: Optional[ScanPublisher] = None
        self._cmd_sub: Optional[CmdVelSubscriber] = None
        self._cmd_pub: Optional[CmdVelPublisher] = None

        self._cmd_vel: Optional[np.ndarray] = None
        self._last_cmd_vel_ts: Optional[float] = None
        self._published_static_tf = False

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
            cmd_vel_in=topics.get("cmd_vel_in", self.cfg.get("cmd_vel_topic", "/cmd_vel")),
            cmd_vel_out=topics.get("cmd_vel_out", "/omninav/cmd_vel"),
        )

    def _resolve_frames(self) -> FrameConfig:
        frames = self.cfg.get("frames", {})
        return FrameConfig(
            map=frames.get("map", "map"),
            odom=frames.get("odom", "odom"),
            base_link=frames.get("base_link", "base_link"),
            laser=frames.get("laser", "laser_frame"),
        )

    def _resolve_publish_switches(self) -> Dict[str, bool]:
        profile_defaults = _PROFILE_DEFAULTS.get(self._profile, _PROFILE_DEFAULTS["all_off"])
        publish_cfg = dict(profile_defaults.get("publish", {}))
        publish_override = self.cfg.get("publish", {})
        for key in ["clock", "tf", "tf_static", "odom", "scan"]:
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

        defaults = {"history": "keep_last", "depth": 10, "reliability": "reliable", "durability": "volatile"}
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

        rclpy.spin_once(self._node, timeout_sec=0)

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

    def _publish_static_tf(self) -> None:
        if self._published_static_tf or self._tf_pub is None or not self._publish.get("tf_static", False):
            return

        from geometry_msgs.msg import TransformStamped

        static_tf = TransformStamped()
        static_tf.header.stamp = self._get_ros_time()
        static_tf.header.frame_id = self._frames.base_link
        static_tf.child_frame_id = self._frames.laser
        static_tf.transform.translation.x = 0.0
        static_tf.transform.translation.y = 0.0
        static_tf.transform.translation.z = 0.0
        static_tf.transform.rotation.w = 1.0
        static_tf.transform.rotation.x = 0.0
        static_tf.transform.rotation.y = 0.0
        static_tf.transform.rotation.z = 0.0
        self._tf_pub.publish_static(static_tf)
        self._published_static_tf = True

    def publish_observation(self, obs: "Observation") -> None:
        if not self._enabled or self._node is None:
            return

        self._publish_clock()
        robot_state = obs.get("robot_state", None)
        stamp = self._get_ros_time()

        if robot_state is not None and self._odom_pub is not None and self._publish.get("odom", False):
            odom_msg = self._odom_pub.publish(robot_state, stamp, self._frames)
            if self._tf_pub is not None and self._publish.get("tf", False):
                odom_tf = self._make_odom_transform(odom_msg.header.stamp, robot_state)
                self._tf_pub.publish_dynamic(odom_tf)

        if self._scan_pub is not None and self._publish.get("scan", False):
            scan_data = Ros2Adapter.pick_scan_data(obs.get("sensors", {}))
            if scan_data is not None:
                self._scan_pub.publish(scan_data, stamp, self._frames.laser)

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
