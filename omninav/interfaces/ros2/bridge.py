"""
ROS2 Bridge for OmniNav

Provides ROS2 integration for publishing sensor data and receiving control commands.
"""

from typing import Optional, Dict, Any, List, TYPE_CHECKING
import numpy as np
from omegaconf import DictConfig

if TYPE_CHECKING:
    from omninav.core.base import SimulationManagerBase
    from omninav.robots.base import RobotBase


class Ros2Bridge:
    """
    ROS2 bridge for OmniNav simulation.
    
    Handles:
    - Publishing sensor data (Lidar, Camera, Odom)
    - Subscribing to control commands (cmd_vel)
    - Clock synchronization
    
    Usage:
        bridge = Ros2Bridge(cfg, sim_manager)
        bridge.setup(robot)
        
        while running:
            sim_manager.step()
            bridge.spin_once()
            cmd_vel = bridge.get_cmd_vel()
    
    Config example:
        ros2:
          enabled: true
          node_name: omninav
          publish_rate: 30.0
          topics:
            scan: /scan
            image: /camera/image_raw
            depth: /camera/depth
            odom: /odom
            cmd_vel: /cmd_vel
    """
    
    def __init__(self, cfg: DictConfig, sim: "SimulationManagerBase"):
        """
        Initialize ROS2 bridge.
        
        Args:
            cfg: Configuration (ros2 section of main config)
            sim: Simulation manager instance
        """
        self.cfg = cfg
        self.sim = sim
        self._enabled = cfg.get("enabled", False)
        
        # ROS2 objects (lazily initialized)
        self._node = None
        self._publishers: Dict[str, Any] = {}
        self._subscribers: Dict[str, Any] = {}
        self._robot: Optional["RobotBase"] = None
        
        # Latest received cmd_vel
        self._cmd_vel: Optional[np.ndarray] = None
        
        if self._enabled:
            self._init_ros2()
    
    def _init_ros2(self) -> None:
        """Initialize ROS2 node and communications."""
        try:
            import rclpy
            from rclpy.node import Node
            from rclpy.qos import QoSProfile, ReliabilityPolicy
            
            if not rclpy.ok():
                rclpy.init()
            
            node_name = self.cfg.get("node_name", "omninav")
            self._node = rclpy.create_node(node_name)
            
            # Setup publishers and subscribers
            self._setup_publishers()
            self._setup_subscribers()
            
        except ImportError:
            import warnings
            warnings.warn(
                "rclpy not found. ROS2 bridge disabled. "
                "Install ROS2 and source setup.bash to enable."
            )
            self._enabled = False
    
    def _setup_publishers(self) -> None:
        """Setup ROS2 publishers."""
        if self._node is None:
            return
        
        from sensor_msgs.msg import LaserScan, Image
        from nav_msgs.msg import Odometry
        from rosgraph_msgs.msg import Clock
        
        topics = self.cfg.get("topics", {})
        qos = 10
        
        # Clock publisher
        self._publishers["clock"] = self._node.create_publisher(
            Clock, "/clock", qos
        )
        
        # Sensor publishers (created when robot is set up)
        # These are placeholders - actual creation happens in setup()
    
    def _setup_subscribers(self) -> None:
        """Setup ROS2 subscribers."""
        if self._node is None:
            return
        
        from geometry_msgs.msg import Twist
        
        topics = self.cfg.get("topics", {})
        cmd_vel_topic = topics.get("cmd_vel", "/cmd_vel")
        
        self._subscribers["cmd_vel"] = self._node.create_subscription(
            Twist,
            cmd_vel_topic,
            self._cmd_vel_callback,
            10
        )
    
    def _cmd_vel_callback(self, msg) -> None:
        """Handle incoming cmd_vel message."""
        self._cmd_vel = np.array([
            msg.linear.x,
            msg.linear.y,
            msg.angular.z
        ], dtype=np.float32)
    
    def setup(self, robot: "RobotBase") -> None:
        """
        Setup bridge with robot reference.
        
        Creates sensor-specific publishers based on robot's mounted sensors.
        
        Args:
            robot: Robot instance with sensors
        """
        self._robot = robot
        
        if not self._enabled or self._node is None:
            return
        
        from sensor_msgs.msg import LaserScan, Image
        from nav_msgs.msg import Odometry
        
        topics = self.cfg.get("topics", {})
        
        # Create Lidar publisher if robot has lidar
        if "lidar_2d" in robot.sensors:
            scan_topic = topics.get("scan", "/scan")
            self._publishers["scan"] = self._node.create_publisher(
                LaserScan, scan_topic, 10
            )
        
        # Create camera publishers if robot has camera
        if "camera" in robot.sensors:
            image_topic = topics.get("image", "/camera/image_raw")
            depth_topic = topics.get("depth", "/camera/depth")
            self._publishers["image"] = self._node.create_publisher(
                Image, image_topic, 10
            )
            self._publishers["depth"] = self._node.create_publisher(
                Image, depth_topic, 10
            )
        
        # Odometry publisher
        odom_topic = topics.get("odom", "/odom")
        self._publishers["odom"] = self._node.create_publisher(
            Odometry, odom_topic, 10
        )
    
    def spin_once(self) -> None:
        """
        Process ROS2 callbacks and publish data.
        
        Should be called after each simulation step.
        """
        if not self._enabled or self._node is None:
            return
        
        import rclpy
        
        # Process incoming messages
        rclpy.spin_once(self._node, timeout_sec=0)
        
        # Publish clock
        self._publish_clock()
        
        # Publish sensor data
        if self._robot is not None:
            self._publish_sensors()
            self._publish_odometry()
    
    def _publish_clock(self) -> None:
        """Publish simulation clock."""
        from rosgraph_msgs.msg import Clock
        from builtin_interfaces.msg import Time
        
        sim_time = self.sim.get_sim_time()
        sec = int(sim_time)
        nanosec = int((sim_time - sec) * 1e9)
        
        msg = Clock()
        msg.clock = Time(sec=sec, nanosec=nanosec)
        self._publishers["clock"].publish(msg)
    
    def _publish_sensors(self) -> None:
        """Publish sensor data."""
        if self._robot is None:
            return
        
        # Publish Lidar
        if "scan" in self._publishers and "lidar_2d" in self._robot.sensors:
            lidar = self._robot.sensors["lidar_2d"]
            if lidar.is_ready:
                data = lidar.get_data()
                self._publish_laser_scan(data)
        
        # Publish Camera
        if "image" in self._publishers and "camera" in self._robot.sensors:
            camera = self._robot.sensors["camera"]
            if camera.is_ready:
                data = camera.get_data()
                self._publish_camera_images(data)
    
    def _publish_laser_scan(self, data: Dict[str, np.ndarray]) -> None:
        """Publish LaserScan message."""
        from sensor_msgs.msg import LaserScan
        
        lidar = self._robot.sensors["lidar_2d"]
        
        msg = LaserScan()
        msg.header.stamp = self._get_ros_time()
        msg.header.frame_id = "laser_frame"
        msg.angle_min = lidar.angle_min
        msg.angle_max = lidar.angle_max
        msg.angle_increment = lidar.angle_increment
        msg.time_increment = 0.0
        msg.scan_time = 0.0
        msg.range_min = lidar._min_range
        msg.range_max = lidar._max_range
        msg.ranges = data["ranges"].tolist()
        
        self._publishers["scan"].publish(msg)
    
    def _publish_camera_images(self, data: Dict[str, np.ndarray]) -> None:
        """Publish camera Image messages."""
        from sensor_msgs.msg import Image
        
        if "rgb" in data and "image" in self._publishers:
            rgb = data["rgb"]
            msg = Image()
            msg.header.stamp = self._get_ros_time()
            msg.header.frame_id = "camera_frame"
            msg.height = rgb.shape[0]
            msg.width = rgb.shape[1]
            msg.encoding = "rgb8"
            msg.is_bigendian = False
            msg.step = rgb.shape[1] * 3
            msg.data = rgb.tobytes()
            self._publishers["image"].publish(msg)
        
        if "depth" in data and "depth" in self._publishers:
            depth = data["depth"]
            msg = Image()
            msg.header.stamp = self._get_ros_time()
            msg.header.frame_id = "camera_frame"
            msg.height = depth.shape[0]
            msg.width = depth.shape[1]
            msg.encoding = "32FC1"
            msg.is_bigendian = False
            msg.step = depth.shape[1] * 4
            msg.data = depth.tobytes()
            self._publishers["depth"].publish(msg)
    
    def _publish_odometry(self) -> None:
        """Publish odometry message."""
        if "odom" not in self._publishers or self._robot is None:
            return
        
        from nav_msgs.msg import Odometry
        
        state = self._robot.get_state()
        
        msg = Odometry()
        msg.header.stamp = self._get_ros_time()
        msg.header.frame_id = "odom"
        msg.child_frame_id = "base_link"
        
        # Position
        msg.pose.pose.position.x = float(state.position[0])
        msg.pose.pose.position.y = float(state.position[1])
        msg.pose.pose.position.z = float(state.position[2])
        
        # Orientation (quaternion)
        msg.pose.pose.orientation.w = float(state.orientation[0])
        msg.pose.pose.orientation.x = float(state.orientation[1])
        msg.pose.pose.orientation.y = float(state.orientation[2])
        msg.pose.pose.orientation.z = float(state.orientation[3])
        
        # Velocity
        msg.twist.twist.linear.x = float(state.linear_velocity[0])
        msg.twist.twist.linear.y = float(state.linear_velocity[1])
        msg.twist.twist.linear.z = float(state.linear_velocity[2])
        msg.twist.twist.angular.x = float(state.angular_velocity[0])
        msg.twist.twist.angular.y = float(state.angular_velocity[1])
        msg.twist.twist.angular.z = float(state.angular_velocity[2])
        
        self._publishers["odom"].publish(msg)
    
    def _get_ros_time(self):
        """Get current ROS time from simulation."""
        from builtin_interfaces.msg import Time
        
        sim_time = self.sim.get_sim_time()
        sec = int(sim_time)
        nanosec = int((sim_time - sec) * 1e9)
        return Time(sec=sec, nanosec=nanosec)
    
    def get_cmd_vel(self) -> Optional[np.ndarray]:
        """
        Get latest cmd_vel from subscriber.
        
        Returns:
            [vx, vy, wz] velocity command or None if no message received
        """
        return self._cmd_vel
    
    def shutdown(self) -> None:
        """Shutdown ROS2 node."""
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
        """Check if ROS2 bridge is enabled."""
        return self._enabled
