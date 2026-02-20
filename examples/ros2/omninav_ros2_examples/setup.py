import os
from pathlib import Path
from setuptools import find_packages, setup

package_name = "omninav_ros2_examples"
this_dir = Path(__file__).resolve().parent
config_root = this_dir / "configs"
config_data_files = []
for cfg_dir in sorted([d for d in config_root.rglob("*") if d.is_dir()] + [config_root]):
    cfg_files = sorted(os.path.relpath(p, this_dir) for p in cfg_dir.glob("*.yaml") if p.is_file())
    if not cfg_files:
        continue
    relative = cfg_dir.relative_to(config_root)
    target = Path("share") / package_name / "omninav_configs" / relative
    config_data_files.append((str(target), cfg_files))

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml", "README.md"]),
        (f"share/{package_name}/launch", [
            "launch/rviz_sensor_demo.launch.py",
            "launch/nav2_full_stack.launch.py",
        ]),
        (f"share/{package_name}/rviz", [
            "rviz/omninav_nav2.rviz",
            "rviz/omninav_sensor.rviz",
        ]),
        (f"share/{package_name}/maps", [
            "maps/complex_flat.yaml",
            "maps/complex_flat.pgm",
        ]),
        (f"share/{package_name}/params", ["params/nav2_params.yaml"]),
        *config_data_files,
    ],
    install_requires=["setuptools"],
    entry_points={
        "console_scripts": [
            "rviz_sensor_demo = omninav_ros2_examples.rviz_sensor_demo:main",
            "nav2_bridge_demo = omninav_ros2_examples.nav2_bridge_demo:main",
        ],
    },
    zip_safe=True,
    maintainer="OmniNav",
    maintainer_email="maintainers@omninav.local",
    description="ROS2 Humble quick-start demos for OmniNav/Nav2 integration",
    license="Apache-2.0",
    tests_require=["pytest"],
)
