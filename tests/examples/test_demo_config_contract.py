"""Contract checks for demo configs."""

from pathlib import Path

from omegaconf import OmegaConf


DEMO_CONFIG_DIR = Path("configs/demo")


def test_demo_configs_exist():
    expected = {
        "teleop_go2.yaml",
        "teleop_go2w.yaml",
        "lidar_visualization.yaml",
        "camera_visualization.yaml",
        "waypoint_navigation.yaml",
        "inspection.yaml",
        "ros2_rviz_sensor.yaml",
        "ros2_nav2_full.yaml",
    }
    existing = {p.name for p in DEMO_CONFIG_DIR.glob("*.yaml")}
    assert expected.issubset(existing)


def test_demo_configs_define_mode():
    for cfg_path in DEMO_CONFIG_DIR.glob("*.yaml"):
        cfg = OmegaConf.load(cfg_path)
        assert "demo" in cfg, f"missing demo section: {cfg_path}"
        assert "mode" in cfg.demo, f"missing demo.mode: {cfg_path}"


def test_examples_do_not_directly_instantiate_core_modules():
    forbidden = (
        "GenesisSimulationManager(",
        "Go2Robot(",
        "Go2wRobot(",
        "WheelController(",
        "KinematicController(",
        "scene.add_entity(",
    )
    for script in Path("examples").glob("*.py"):
        text = script.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden), f"direct core instantiation found in {script}"


def test_examples_do_not_use_demo_runner():
    for script in Path("examples").glob("*.py"):
        text = script.read_text(encoding="utf-8")
        assert "demo_runner" not in text, f"demo_runner import found in {script}"


def test_examples_use_env_config_instantiation():
    for script in Path("examples").glob("*.py"):
        text = script.read_text(encoding="utf-8")
        assert ("OmniNavEnv(" in text) or ("OmniNavEnv.from_config(" in text), (
            f"OmniNavEnv config-based instantiation missing in {script}"
        )
