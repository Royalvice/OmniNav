from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = Path(get_package_share_directory("omninav_ros2_examples"))
    nav2_share = Path(get_package_share_directory("nav2_bringup"))

    map_yaml = LaunchConfiguration("map")
    params_file = LaunchConfiguration("params_file")
    use_sim_time = LaunchConfiguration("use_sim_time")

    omninav_node = Node(
        package="omninav_ros2_examples",
        executable="nav2_bridge_demo",
        output="screen",
    )

    nav2_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(nav2_share / "launch" / "bringup_launch.py")),
        launch_arguments={
            "map": map_yaml,
            "use_sim_time": use_sim_time,
            "autostart": "True",
            "params_file": params_file,
            "slam": "False",
            "use_composition": "False",
            "use_respawn": "False",
        }.items(),
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        output="screen",
        arguments=["-d", str(pkg_share / "rviz" / "omninav_nav2.rviz")],
        parameters=[{"use_sim_time": use_sim_time}],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "map",
                default_value=str(pkg_share / "maps" / "complex_flat.yaml"),
                description="Static occupancy map for Nav2",
            ),
            DeclareLaunchArgument(
                "params_file",
                default_value=str(pkg_share / "params" / "nav2_params.yaml"),
                description="Nav2 parameters file",
            ),
            DeclareLaunchArgument("use_sim_time", default_value="True"),
            omninav_node,
            nav2_bringup,
            rviz_node,
        ]
    )
