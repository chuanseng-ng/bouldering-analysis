"""
Configuration loader for the bouldering analysis application.

This module provides utilities to load and cache application configuration
from YAML files. It handles path resolution relative to the project root
and includes comprehensive error handling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import logging
import threading

logger = logging.getLogger(__name__)

# Cache for configuration to avoid repeated file reads
_config_cache: Optional[dict[str, Any]] = None

# Lock for thread-safe access to the configuration cache
_config_lock = threading.Lock()

# Project root directory (parent of src/)
PROJECT_ROOT = Path(__file__).parent.parent


class ConfigurationError(Exception):
    """Raised when there are issues with configuration loading or validation."""


def get_project_root() -> Path:
    """
    Get the project root directory.

    Returns:
        Path: The absolute path to the project root directory.
    """
    return PROJECT_ROOT


def resolve_path(path_str: str, relative_to: Optional[Path] = None) -> Path:
    """
    Resolve a path string to an absolute path.

    If the path is relative, it will be resolved relative to the project root
    (or a specified directory). Absolute paths are returned as-is.

    Args:
        path_str: The path string to resolve.
        relative_to: Optional base directory for relative paths.
                     Defaults to project root.

    Returns:
        Path: The resolved absolute path.

    Examples:
        >>> resolve_path('data/uploads/')
        Path('/path/to/project/data/uploads')
        >>> resolve_path('/absolute/path')
        Path('/absolute/path')
    """
    path = Path(path_str)

    if path.is_absolute():
        return path

    base_dir = relative_to if relative_to is not None else PROJECT_ROOT
    return (base_dir / path).resolve()


def load_config(
    config_path: str = "src/cfg/user_config.yaml", force_reload: bool = False
) -> dict[str, Any]:
    """
    Load configuration from a YAML file with caching.

    The configuration is cached after the first load to avoid repeated file reads.
    Use force_reload=True to bypass the cache and reload from disk.

    Args:
        config_path: Path to the configuration YAML file, relative to project root.
                     Defaults to 'src/cfg/user_config.yaml'.
        force_reload: If True, bypass the cache and reload from disk.

    Returns:
        Dict[str, Any]: The parsed configuration dictionary.

    Raises:
        ConfigurationError: If the configuration file cannot be found, read,
                           or parsed.

    Examples:
        >>> config = load_config()
        >>> threshold = config['model_defaults']['hold_detection_confidence_threshold']
        >>> print(threshold)
        0.25
    """
    global _config_cache  # pylint: disable=global-statement

    # Thread-safe cache check and update
    with _config_lock:
        # Return cached config if available and not forcing reload
        if _config_cache is not None and not force_reload:
            logger.debug("Returning cached configuration")
            return _config_cache

        try:
            import yaml  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ConfigurationError(
                "PyYAML is required for configuration loading. "
                "Please install it with: pip install PyYAML"
            ) from exc

        # Resolve the config file path
        config_file = resolve_path(config_path)

        # Check if file exists
        if not config_file.exists():
            raise ConfigurationError(
                f"Configuration file not found: {config_file}\n"
                f"Expected location: {config_path} (relative to project root)"
            )

        # Load and parse the YAML file
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            if config is None:
                raise ConfigurationError(f"Configuration file is empty: {config_file}")

            if not isinstance(config, dict):
                raise ConfigurationError(  # pragma: no cover
                    f"Configuration must be a dictionary, got {type(config).__name__}"
                )

            # Validate required configuration sections
            _validate_config(config)

            # Cache the configuration
            _config_cache = config
            logger.info("Configuration loaded successfully from %s", config_file)

            return config

        except yaml.YAMLError as exc:
            raise ConfigurationError(
                f"Error parsing YAML configuration file {config_file}: {exc}"
            ) from exc
        except (OSError, IOError) as exc:
            raise ConfigurationError(  # pragma: no cover
                f"Error reading configuration file {config_file}: {exc}"
            ) from exc


def _validate_config(config: dict[str, Any]) -> None:
    """
    Validate the structure of the configuration dictionary.

    Ensures that required top-level sections and keys exist.

    Args:
        config: The configuration dictionary to validate.

    Raises:
        ConfigurationError: If required configuration sections or keys are missing.
    """
    required_sections = {
        "model_defaults": ["hold_detection_confidence_threshold"],
        "model_paths": ["base_yolov8", "fine_tuned_models"],
        "data_paths": ["hold_dataset", "uploads"],
    }

    for section, keys in required_sections.items():
        if section not in config:
            raise ConfigurationError(
                f"Missing required configuration section: '{section}'"
            )

        if not isinstance(config[section], dict):
            raise ConfigurationError(
                f"Configuration section '{section}' must be a dictionary"
            )

        for key in keys:
            if key not in config[section]:
                raise ConfigurationError(
                    f"Missing required configuration key: '{section}.{key}'"
                )


def get_config_value(key_path: str, default: Any = None) -> Any:
    """
    Get a configuration value using dot notation.

    Args:
        key_path: Dot-separated path to the configuration value.
                  Example: 'model_defaults.hold_detection_confidence_threshold'
        default: Default value to return if the key is not found.

    Returns:
        The configuration value or the default if not found.

    Examples:
        >>> threshold = get_config_value('model_defaults.hold_detection_confidence_threshold')
        >>> print(threshold)
        0.25
        >>> uploads = get_config_value('data_paths.uploads')
        >>> print(uploads)
        data/uploads/
    """
    config = load_config()
    keys = key_path.split(".")

    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default

    return value


def clear_config_cache() -> None:
    """
    Clear the cached configuration.

    This forces the next call to load_config() to reload from disk.
    Useful for testing or when configuration files are modified at runtime.
    """
    global _config_cache  # pylint: disable=global-statement
    with _config_lock:
        _config_cache = None
        logger.debug("Configuration cache cleared")


def _get_path_from_section(section: str, path_key: str) -> Path:
    """
    Get a resolved absolute path from a specific configuration section.

    Args:
        section: The configuration section name (e.g., 'model_paths', 'data_paths').
        path_key: The configuration key for the path within the section.

    Returns:
        Path: The resolved absolute path.

    Raises:
        ConfigurationError: If the section or path key is not found in configuration.
    """
    config = load_config()

    if section not in config:
        raise ConfigurationError(f"Missing '{section}' section in configuration")

    if path_key not in config[section]:
        raise ConfigurationError(f"Path '{path_key}' not found in '{section}' section")

    path_str = config[section][path_key]
    return resolve_path(path_str)


def get_model_path(path_key: str) -> Path:
    """
    Get a resolved absolute path for a model-related path from configuration.

    Args:
        path_key: The configuration key for the model path.
                  Example: 'base_yolov8' or 'fine_tuned_models'

    Returns:
        Path: The resolved absolute path.

    Raises:
        ConfigurationError: If the path key is not found in configuration.

    Examples:
        >>> base_model = get_model_path('base_yolov8')
        >>> print(base_model)
        /path/to/project/yolov8n.pt
    """
    try:
        return _get_path_from_section("model_paths", path_key)
    except ConfigurationError as exc:
        # Re-raise with more specific message if it's about a missing path key
        if "not found in" in str(exc):
            raise ConfigurationError(
                f"Model path '{path_key}' not found in configuration"
            ) from exc
        raise


def get_data_path(path_key: str) -> Path:
    """
    Get a resolved absolute path for a data-related path from configuration.

    Args:
        path_key: The configuration key for the data path.
                  Example: 'hold_dataset' or 'uploads'

    Returns:
        Path: The resolved absolute path.

    Raises:
        ConfigurationError: If the path key is not found in configuration.

    Examples:
        >>> uploads_dir = get_data_path('uploads')
        >>> print(uploads_dir)
        /path/to/project/data/uploads/
    """
    try:
        return _get_path_from_section("data_paths", path_key)
    except ConfigurationError as exc:
        # Re-raise with more specific message if it's about a missing path key
        if "not found in" in str(exc):
            raise ConfigurationError(
                f"Data path '{path_key}' not found in configuration"
            ) from exc
        raise
