"""
Unit tests for src/config.py - Configuration loading and management.

Tests cover:
- Configuration file loading and caching
- Path resolution
- Configuration value retrieval with dot notation
- Error handling for missing/invalid configurations
- Cache management
"""

from pathlib import Path

import pytest

from src.config import (
    load_config,
    get_config_value,
    get_model_path,
    get_data_path,
    clear_config_cache,
    resolve_path,
    get_project_root,
    ConfigurationError,
)


class TestGetProjectRoot:  # pylint: disable=too-few-public-methods
    """Test cases for get_project_root function."""

    def test_get_project_root_returns_path(self):
        """Test that get_project_root returns a Path object."""
        root = get_project_root()
        assert isinstance(root, Path)
        assert root.exists()
        assert root.is_dir()


class TestResolvePath:
    """Test cases for resolve_path function."""

    def test_resolve_relative_path(self):
        """Test resolving a relative path."""
        result = resolve_path("data/uploads")
        assert isinstance(result, Path)
        assert result.is_absolute()
        assert "data" in str(result)
        assert "uploads" in str(result)

    def test_resolve_absolute_path(self):
        """Test that absolute paths are returned as-is."""
        # Use a platform-appropriate absolute path
        import platform  # pylint: disable=import-outside-toplevel

        if platform.system() == "Windows":
            abs_path = Path("C:/absolute/path/to/file")
        else:
            abs_path = Path("/absolute/path/to/file")
        result = resolve_path(str(abs_path))
        assert result.is_absolute()
        assert "absolute" in str(result)

    def test_resolve_path_with_custom_base(self, tmp_path):
        """Test resolving a path with a custom base directory."""
        result = resolve_path("subdir/file.txt", relative_to=tmp_path)
        assert str(tmp_path) in str(result)
        assert "subdir" in str(result)


class TestLoadConfig:
    """Test cases for load_config function."""

    def test_load_config_success(self, test_config_yaml, monkeypatch):  # pylint: disable=unused-argument
        """Test successful configuration loading."""
        # Clear cache before test
        clear_config_cache()

        # Mock resolve_path to return our test config
        def mock_resolve(path):
            return test_config_yaml

        monkeypatch.setattr("src.config.resolve_path", mock_resolve)

        config = load_config()

        assert isinstance(config, dict)
        assert "model_defaults" in config
        assert "model_paths" in config
        assert "data_paths" in config
        assert config["model_defaults"]["hold_detection_confidence_threshold"] == 0.25

    def test_load_config_caching(self, test_config_yaml, monkeypatch):  # pylint: disable=unused-argument
        """Test that configuration is cached after first load."""
        clear_config_cache()

        def mock_resolve(path):
            return test_config_yaml

        monkeypatch.setattr("src.config.resolve_path", mock_resolve)

        # First load
        config1 = load_config()

        # Second load should return cached version
        config2 = load_config()

        # Should be the same object (cached)
        assert config1 is config2

    def test_load_config_force_reload(self, test_config_yaml, monkeypatch):  # pylint: disable=unused-argument
        """Test force reload bypasses cache."""
        clear_config_cache()

        def mock_resolve(path):
            return test_config_yaml

        monkeypatch.setattr("src.config.resolve_path", mock_resolve)

        # First load
        config1 = load_config()

        # Force reload
        config2 = load_config(force_reload=True)

        # Should have same content but potentially different objects
        assert config1 == config2

    def test_load_config_file_not_found(self, tmp_path, monkeypatch):  # pylint: disable=unused-argument
        """Test error handling when config file doesn't exist."""
        clear_config_cache()

        nonexistent_file = tmp_path / "nonexistent.yaml"

        def mock_resolve(path):
            return nonexistent_file

        monkeypatch.setattr("src.config.resolve_path", mock_resolve)

        with pytest.raises(ConfigurationError, match="Configuration file not found"):
            load_config()

    def test_load_config_invalid_yaml(self, invalid_config_yaml, monkeypatch):  # pylint: disable=unused-argument
        """Test error handling for invalid YAML syntax."""
        clear_config_cache()

        def mock_resolve(path):
            return invalid_config_yaml

        monkeypatch.setattr("src.config.resolve_path", mock_resolve)

        with pytest.raises(ConfigurationError, match="Error parsing YAML"):
            load_config()

    def test_load_config_empty_file(self, empty_config_yaml, monkeypatch):  # pylint: disable=unused-argument
        """Test error handling for empty configuration file."""
        clear_config_cache()

        def mock_resolve(path):
            return empty_config_yaml

        monkeypatch.setattr("src.config.resolve_path", mock_resolve)

        with pytest.raises(ConfigurationError, match="Configuration file is empty"):
            load_config()

    def test_load_config_missing_required_sections(self, tmp_path, monkeypatch):  # pylint: disable=unused-argument
        """Test validation of required configuration sections."""
        clear_config_cache()

        # Create config with missing sections
        incomplete_config = tmp_path / "incomplete.yaml"
        incomplete_config.write_text("model_defaults:\n  threshold: 0.5\n")

        def mock_resolve(path):
            return incomplete_config

        monkeypatch.setattr("src.config.resolve_path", mock_resolve)

        with pytest.raises(
            ConfigurationError, match="Missing required configuration section"
        ):
            load_config()

    def test_load_config_missing_yaml_library(self, monkeypatch):  # pylint: disable=unused-argument
        """Test error handling when PyYAML is not installed."""
        clear_config_cache()

        # Mock import error for yaml
        import builtins  # pylint: disable=import-outside-toplevel

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "yaml":
                raise ImportError("No module named 'yaml'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        with pytest.raises(ConfigurationError, match="PyYAML is required"):
            load_config()


class TestGetConfigValue:
    """Test cases for get_config_value function."""

    def test_get_config_value_existing_key(self, test_config_yaml, monkeypatch):  # pylint: disable=unused-argument
        """Test retrieving an existing configuration value."""
        clear_config_cache()

        def mock_resolve(path):
            return test_config_yaml

        monkeypatch.setattr("src.config.resolve_path", mock_resolve)

        value = get_config_value("model_defaults.hold_detection_confidence_threshold")
        assert value == 0.25

    def test_get_config_value_nested_key(self, test_config_yaml, monkeypatch):  # pylint: disable=unused-argument
        """Test retrieving a nested configuration value."""
        clear_config_cache()

        def mock_resolve(path):
            return test_config_yaml

        monkeypatch.setattr("src.config.resolve_path", mock_resolve)

        value = get_config_value("model_paths.base_yolov8")
        assert value == "yolov8n.pt"

    def test_get_config_value_missing_key_with_default(
        self, test_config_yaml, monkeypatch
    ):  # pylint: disable=unused-argument
        """Test default value is returned for missing keys."""
        clear_config_cache()

        def mock_resolve(path):
            return test_config_yaml

        monkeypatch.setattr("src.config.resolve_path", mock_resolve)

        value = get_config_value("nonexistent.key", default="default_value")
        assert value == "default_value"

    def test_get_config_value_missing_key_no_default(
        self, test_config_yaml, monkeypatch
    ):  # pylint: disable=unused-argument
        """Test None is returned for missing keys without default."""
        clear_config_cache()

        def mock_resolve(path):
            return test_config_yaml

        monkeypatch.setattr("src.config.resolve_path", mock_resolve)

        value = get_config_value("nonexistent.key")
        assert value is None


class TestClearConfigCache:  # pylint: disable=too-few-public-methods
    """Test cases for clear_config_cache function."""

    def test_clear_config_cache(self, test_config_yaml, monkeypatch):  # pylint: disable=unused-argument
        """Test that cache is cleared correctly."""
        clear_config_cache()

        def mock_resolve(path):
            return test_config_yaml

        monkeypatch.setattr("src.config.resolve_path", mock_resolve)

        # Load config
        config1 = load_config()

        # Clear cache
        clear_config_cache()

        # Load again - should reload from file
        config2 = load_config()

        # Should have same content
        assert config1 == config2


class TestGetModelPath:
    """Test cases for get_model_path function."""

    def test_get_model_path_success(self, test_config_yaml, monkeypatch):  # pylint: disable=unused-argument
        """Test successful retrieval of model path."""
        clear_config_cache()

        # Need to mock resolve_path to return test config on first call,
        # then return the actual path on second call
        call_count = [0]
        original_resolve = resolve_path

        def mock_resolve(path):
            call_count[0] += 1
            if call_count[0] == 1:
                return test_config_yaml
            # For the model path itself, use original resolve
            return original_resolve(path)

        monkeypatch.setattr("src.config.resolve_path", mock_resolve)

        path = get_model_path("base_yolov8")
        assert isinstance(path, Path)
        assert path.is_absolute()

    def test_get_model_path_missing_key(self, test_config_yaml, monkeypatch):  # pylint: disable=unused-argument
        """Test error handling for missing model path key."""
        clear_config_cache()

        def mock_resolve(path):
            return test_config_yaml

        monkeypatch.setattr("src.config.resolve_path", mock_resolve)

        with pytest.raises(ConfigurationError, match="Model path .* not found"):
            get_model_path("nonexistent_model")

    def test_get_model_path_missing_section(self, tmp_path, monkeypatch):  # pylint: disable=unused-argument
        """Test error handling when model_paths section is missing."""
        clear_config_cache()

        # Create config without model_paths - must have all required sections
        config_file = tmp_path / "no_model_paths.yaml"
        import yaml  # pylint: disable=import-outside-toplevel

        config_data = {
            "model_defaults": {"hold_detection_confidence_threshold": 0.25},
            "data_paths": {"hold_dataset": "data/", "uploads": "uploads/"},
        }
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        def mock_resolve(path):
            return config_file

        monkeypatch.setattr("src.config.resolve_path", mock_resolve)

        with pytest.raises(ConfigurationError, match="Missing required configuration"):
            get_model_path("base_yolov8")


class TestGetDataPath:
    """Test cases for get_data_path function."""

    def test_get_data_path_success(self, test_config_yaml, monkeypatch):  # pylint: disable=unused-argument
        """Test successful retrieval of data path."""
        clear_config_cache()

        call_count = [0]
        original_resolve = resolve_path

        def mock_resolve(path):
            call_count[0] += 1
            if call_count[0] == 1:
                return test_config_yaml
            return original_resolve(path)

        monkeypatch.setattr("src.config.resolve_path", mock_resolve)

        path = get_data_path("uploads")
        assert isinstance(path, Path)
        assert path.is_absolute()

    def test_get_data_path_missing_key(self, test_config_yaml, monkeypatch):  # pylint: disable=unused-argument
        """Test error handling for missing data path key."""
        clear_config_cache()

        def mock_resolve(path):
            return test_config_yaml

        monkeypatch.setattr("src.config.resolve_path", mock_resolve)

        with pytest.raises(ConfigurationError, match="Data path .* not found"):
            get_data_path("nonexistent_data")

    def test_get_data_path_missing_section(self, tmp_path, monkeypatch):  # pylint: disable=unused-argument
        """Test error handling when data_paths section is missing."""
        clear_config_cache()

        # Create config without data_paths - must have all required sections
        config_file = tmp_path / "no_data_paths.yaml"
        import yaml  # pylint: disable=import-outside-toplevel

        config_data = {
            "model_defaults": {"hold_detection_confidence_threshold": 0.25},
            "model_paths": {
                "base_yolov8": "yolov8n.pt",
                "fine_tuned_models": "models/",
            },
        }
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config_data, f)

        def mock_resolve(path):
            return config_file

        monkeypatch.setattr("src.config.resolve_path", mock_resolve)

        with pytest.raises(ConfigurationError, match="Missing required configuration"):
            get_data_path("uploads")
