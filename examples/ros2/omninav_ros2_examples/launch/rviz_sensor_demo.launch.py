from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


def generate_launch_description():
    pkg_share = Path(get_package_share_directory("omninav_ros2_examples"))
    rviz_cfg = pkg_share / "rviz" / "omninav_sensor.rviz"

    return LaunchDescription(
        [
            Node(
                package="omninav_ros2_examples",
                executable="rviz_sensor_demo",
                arguments=["--show-viewer"],
                output="screen",
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                output="screen",
                arguments=["-d", str(rviz_cfg)],
                parameters=[{"use_sim_time": True}],
            ),
        ]
    )
