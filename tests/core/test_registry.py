"""
Tests for Registry mechanism.
"""

import pytest
from omegaconf import OmegaConf

from omninav.core.registry import Registry, ROBOT_REGISTRY, SENSOR_REGISTRY


class TestRegistry:
    """Test suite for Registry class."""
    
    def test_register_and_get(self):
        """Test registering a class and retrieving it."""
        registry = Registry("test")
        
        @registry.register("my_class")
        class MyClass:
            pass
        
        assert "my_class" in registry
        assert registry.get("my_class") is MyClass
    
    def test_register_default_name(self):
        """Test registration with default class name."""
        registry = Registry("test")
        
        @registry.register()
        class AnotherClass:
            pass
        
        assert "AnotherClass" in registry
    
    def test_duplicate_registration_raises(self):
        """Test that duplicate registration raises KeyError."""
        registry = Registry("test")
        
        @registry.register("duplicate")
        class First:
            pass
        
        with pytest.raises(KeyError, match="already registered"):
            @registry.register("duplicate")
            class Second:
                pass
    
    def test_get_missing_raises(self):
        """Test that getting unregistered name raises KeyError."""
        registry = Registry("test")
        
        with pytest.raises(KeyError, match="not registered"):
            registry.get("nonexistent")
    
    def test_build_from_config(self):
        """Test building instance from configuration."""
        registry = Registry("test")
        
        @registry.register("buildable")
        class Buildable:
            def __init__(self, cfg):
                self.cfg = cfg
                self.value = cfg.get("value", 0)
        
        cfg = OmegaConf.create({
            "type": "buildable",
            "value": 42,
        })
        
        instance = registry.build(cfg)
        assert isinstance(instance, Buildable)
        assert instance.value == 42
    
    def test_build_missing_type_raises(self):
        """Test that build without type field raises KeyError."""
        registry = Registry("test")
        
        cfg = OmegaConf.create({"value": 42})
        
        with pytest.raises(KeyError, match="must contain 'type'"):
            registry.build(cfg)
    
    def test_build_with_kwargs(self):
        """Test build passing additional kwargs."""
        registry = Registry("test")
        
        @registry.register("with_kwargs")
        class WithKwargs:
            def __init__(self, cfg, extra=None):
                self.extra = extra
        
        cfg = OmegaConf.create({"type": "with_kwargs"})
        instance = registry.build(cfg, extra="test_value")
        assert instance.extra == "test_value"
    
    def test_registered_names(self):
        """Test getting list of registered names."""
        registry = Registry("test")
        
        @registry.register("first")
        class First:
            pass
        
        @registry.register("second")
        class Second:
            pass
        
        names = registry.registered_names
        assert "first" in names
        assert "second" in names
    
    def test_global_registries_exist(self):
        """Test that global registries are properly initialized."""
        assert ROBOT_REGISTRY.name == "robot"
        assert SENSOR_REGISTRY.name == "sensor"
