"""Composable ROS2 publisher/subscriber building blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TopicConfig:
    clock: str
    tf: str
    tf_static: str
    odom: str
    scan: str
    rgb_image: str
    depth_image: str
    rgb_camera_info: str
    depth_camera_info: str
    cmd_vel_in: str
    cmd_vel_out: str


@dataclass
class FrameConfig:
    map: str
    odom: str
    base_link: str
    lidar: str
    rgb_camera: str
    depth_camera: str


class ClockPublisher:
    def __init__(self, node, topic: str, qos: Any):
        from rosgraph_msgs.msg import Clock

        self._Clock = Clock
        self._publisher = node.create_publisher(Clock, topic, qos)

    def publish(self, stamp):
        msg = self._Clock()
        msg.clock = stamp
        self._publisher.publish(msg)


class TfPublisher:
    def __init__(self, node, topic_tf: str, topic_tf_static: str, qos_tf: Any, qos_tf_static: Any):
        from tf2_msgs.msg import TFMessage

        self._TFMessage = TFMessage
        self._pub_tf = node.create_publisher(TFMessage, topic_tf, qos_tf)
        self._pub_tf_static = node.create_publisher(TFMessage, topic_tf_static, qos_tf_static)

    def publish_dynamic(self, transform):
        msg = self._TFMessage()
        msg.transforms.append(transform)
        self._pub_tf.publish(msg)

    def publish_static(self, transform):
        msg = self._TFMessage()
        msg.transforms.append(transform)
        self._pub_tf_static.publish(msg)


class OdomPublisher:
    def __init__(self, node, topic: str, qos: Any):
        from nav_msgs.msg import Odometry

        self._Odometry = Odometry
        self._publisher = node.create_publisher(Odometry, topic, qos)

    def publish(self, state: dict, stamp, frames: FrameConfig):
        pos = np.asarray(state.get("position", np.zeros((1, 3), dtype=np.float32)))
        orient = np.asarray(state.get("orientation", np.array([[1, 0, 0, 0]], dtype=np.float32)))
        lin_vel = np.asarray(state.get("linear_velocity", np.zeros((1, 3), dtype=np.float32)))
        ang_vel = np.asarray(state.get("angular_velocity", np.zeros((1, 3), dtype=np.float32)))
        if pos.ndim == 2:
            pos = pos[0]
        if orient.ndim == 2:
            orient = orient[0]
        if lin_vel.ndim == 2:
            lin_vel = lin_vel[0]
        if ang_vel.ndim == 2:
            ang_vel = ang_vel[0]

        msg = self._Odometry()
        msg.header.stamp = stamp
        msg.header.frame_id = frames.odom
        msg.child_frame_id = frames.base_link

        msg.pose.pose.position.x = float(pos[0])
        msg.pose.pose.position.y = float(pos[1])
        msg.pose.pose.position.z = float(pos[2])

        msg.pose.pose.orientation.w = float(orient[0])
        msg.pose.pose.orientation.x = float(orient[1])
        msg.pose.pose.orientation.y = float(orient[2])
        msg.pose.pose.orientation.z = float(orient[3])

        msg.twist.twist.linear.x = float(lin_vel[0])
        msg.twist.twist.linear.y = float(lin_vel[1])
        msg.twist.twist.linear.z = float(lin_vel[2])
        msg.twist.twist.angular.x = float(ang_vel[0])
        msg.twist.twist.angular.y = float(ang_vel[1])
        msg.twist.twist.angular.z = float(ang_vel[2])

        self._publisher.publish(msg)
        return msg


class ScanPublisher:
    def __init__(self, node, topic: str, qos: Any):
        from sensor_msgs.msg import LaserScan

        self._LaserScan = LaserScan
        self._publisher = node.create_publisher(LaserScan, topic, qos)

    def publish(self, data: dict, stamp, frame_id: str):
        ranges = np.asarray(data.get("ranges", np.array([], dtype=np.float32)), dtype=np.float32).reshape(-1)
        msg = self._LaserScan()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.angle_min = float(data.get("angle_min", -np.pi))
        msg.angle_max = float(data.get("angle_max", np.pi))
        msg.angle_increment = float(data.get("angle_increment", 0.0))
        msg.time_increment = float(data.get("time_increment", 0.0))
        msg.scan_time = float(data.get("scan_time", 0.0))
        msg.range_min = float(data.get("range_min", 0.1))
        msg.range_max = float(data.get("range_max", 30.0))
        msg.ranges = ranges.tolist()

        self._publisher.publish(msg)


class ImagePublisher:
    def __init__(self, node, topic: str, qos: Any):
        from sensor_msgs.msg import Image

        self._Image = Image
        self._publisher = node.create_publisher(Image, topic, qos)

    def publish_rgb8(self, image: np.ndarray, stamp, frame_id: str):
        height, width = image.shape[0], image.shape[1]
        msg = self._Image()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height = int(height)
        msg.width = int(width)
        msg.encoding = "rgb8"
        msg.is_bigendian = 0
        msg.step = int(width * 3)
        msg.data = image.tobytes()
        self._publisher.publish(msg)

    def publish_depth32f(self, image: np.ndarray, stamp, frame_id: str):
        height, width = image.shape[0], image.shape[1]
        msg = self._Image()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height = int(height)
        msg.width = int(width)
        msg.encoding = "32FC1"
        msg.is_bigendian = 0
        msg.step = int(width * 4)
        msg.data = image.tobytes()
        self._publisher.publish(msg)


class CameraInfoPublisher:
    def __init__(self, node, topic: str, qos: Any):
        from sensor_msgs.msg import CameraInfo

        self._CameraInfo = CameraInfo
        self._publisher = node.create_publisher(CameraInfo, topic, qos)

    def publish(self, height: int, width: int, stamp, frame_id: str, fov_deg: float):
        fx = (width / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0) if fov_deg > 0 else float(width)
        fy = fx
        cx = width / 2.0
        cy = height / 2.0

        msg = self._CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.height = int(height)
        msg.width = int(width)
        msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        self._publisher.publish(msg)


class CmdVelSubscriber:
    def __init__(self, node, topic: str, qos: Any, callback):
        from geometry_msgs.msg import Twist

        self._subscriber = node.create_subscription(Twist, topic, callback, qos)


class CmdVelPublisher:
    def __init__(self, node, topic: str, qos: Any):
        from geometry_msgs.msg import Twist

        self._Twist = Twist
        self._publisher = node.create_publisher(Twist, topic, qos)

    def publish(self, cmd_vel: np.ndarray):
        msg = self._Twist()
        msg.linear.x = float(cmd_vel[0])
        msg.linear.y = float(cmd_vel[1])
        msg.angular.z = float(cmd_vel[2])
        self._publisher.publish(msg)
