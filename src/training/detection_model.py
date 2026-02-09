"""YOLOv8 detection model definition for hold/volume detection.

This module provides functions to build and configure YOLOv8m models
for detecting climbing holds and volumes in bouldering route images.

The model uses a fixed input resolution of 640x640 and outputs bounding
boxes with class predictions for two classes: hold and volume.

Example:
    >>> from src.training.detection_model import build_hold_detector
    >>> model = build_hold_detector()
    >>> print(model.names)
    {0: 'hold', 1: 'volume'}
"""

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator
from ultralytics import YOLO

from src.logging_config import get_logger
from src.training.datasets import EXPECTED_CLASSES

logger = get_logger(__name__)

# Model architecture configuration
DEFAULT_MODEL_SIZE = "yolov8m"  # Medium model for balance of speed and accuracy
INPUT_RESOLUTION = 640  # Standard YOLOv8 resolution (640x640)
VALID_MODEL_SIZES = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]


class DetectionHyperparameters(BaseModel):
    """Hyperparameters for hold detection model training.

    These parameters control the training process, data augmentation,
    and optimization settings for the YOLOv8 detection model.

    Attributes:
        epochs: Number of training epochs.
        batch_size: Batch size for training. Use -1 for auto-batch.
        image_size: Input image size (square). Should be 640 for standard YOLOv8.
        learning_rate: Initial learning rate (Adam optimizer).
        weight_decay: L2 regularization weight.
        patience: Early stopping patience (epochs without improvement).
        optimizer: Optimizer algorithm (adam, adamw, sgd).
        momentum: Momentum factor for SGD optimizer.
        augment: Enable data augmentation during training.
        hsv_h: Hue augmentation range (0.0-1.0).
        hsv_s: Saturation augmentation range (0.0-1.0).
        hsv_v: Value (brightness) augmentation range (0.0-1.0).
        degrees: Rotation augmentation (±degrees).
        translate: Translation augmentation (fraction of image).
        scale: Scale augmentation (±fraction).
        shear: Shear augmentation (±degrees).
        perspective: Perspective augmentation (0.0-0.001).
        flipud: Probability of vertical flip.
        fliplr: Probability of horizontal flip.
        mosaic: Probability of mosaic augmentation.
        mixup: Probability of mixup augmentation.
        device: Device to use for training (cuda, cpu, or device id).
        workers: Number of data loading workers.
        pretrained: Use COCO pretrained weights as initialization.
        verbose: Enable verbose logging during training.
        seed: Random seed for reproducibility.
    """

    # Pydantic configuration
    model_config = ConfigDict(populate_by_name=True)

    # Training schedule
    epochs: int = Field(default=100, ge=1, le=1000)
    batch_size: int = Field(default=16, ge=-1, alias="batch")  # -1 for auto-batch
    image_size: int = Field(default=INPUT_RESOLUTION, ge=32, alias="imgsz")
    learning_rate: float = Field(default=0.01, gt=0.0, le=1.0, alias="lr0")
    weight_decay: float = Field(default=0.0005, ge=0.0, le=1.0)
    patience: int = Field(default=50, ge=0)

    # Optimizer settings
    optimizer: str = Field(default="AdamW")
    momentum: float = Field(default=0.937, ge=0.0, le=1.0)

    # Augmentation settings
    augment: bool = Field(default=True)
    hsv_h: float = Field(default=0.015, ge=0.0, le=1.0)
    hsv_s: float = Field(default=0.7, ge=0.0, le=1.0)
    hsv_v: float = Field(default=0.4, ge=0.0, le=1.0)
    degrees: float = Field(default=10.0, ge=0.0, le=180.0)
    translate: float = Field(default=0.1, ge=0.0, le=1.0)
    scale: float = Field(default=0.5, ge=0.0, le=1.0)
    shear: float = Field(default=2.0, ge=0.0, le=45.0)
    perspective: float = Field(default=0.0001, ge=0.0, le=0.001)
    flipud: float = Field(default=0.0, ge=0.0, le=1.0)
    fliplr: float = Field(default=0.5, ge=0.0, le=1.0)
    mosaic: float = Field(default=1.0, ge=0.0, le=1.0)
    mixup: float = Field(default=0.0, ge=0.0, le=1.0)

    # Hardware and system
    device: str | int = Field(default="")  # Empty string for auto-detect
    workers: int = Field(default=8, ge=0)
    pretrained: bool = Field(default=True)
    verbose: bool = Field(default=True)
    seed: int = Field(default=0, ge=0)

    @field_validator("batch_size")
    @classmethod
    def validate_batch_size(cls, v: int) -> int:
        """Validate batch_size is -1 (auto) or a positive integer."""
        if v == 0 or not (v == -1 or v > 0):
            raise ValueError("batch_size must be -1 (auto) or a positive integer")
        return v

    @field_validator("optimizer")
    @classmethod
    def validate_optimizer(cls, v: str) -> str:
        """Validate optimizer is supported by YOLOv8."""
        valid_optimizers = {"SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"}
        if v not in valid_optimizers:
            raise ValueError(f"optimizer must be one of {valid_optimizers}, got '{v}'")
        return v

    @field_validator("image_size")
    @classmethod
    def validate_image_size(cls, v: int) -> int:
        """Validate image size is a multiple of 32 (YOLO requirement)."""
        if v % 32 != 0:
            raise ValueError(f"image_size must be a multiple of 32, got {v}")
        return v

    def to_dict(self) -> dict[str, Any]:
        """Convert hyperparameters to dictionary for YOLO training.

        Returns:
            Dictionary with hyperparameter names and values suitable
            for passing to YOLO.train() method.
        """
        # Use model_dump to get the dictionary representation
        # by_alias=True ensures we use aliases like 'lr0' instead of 'learning_rate'
        return self.model_dump(by_alias=True, exclude_none=True)


def build_hold_detector(
    model_size: str = DEFAULT_MODEL_SIZE,
    pretrained: bool = True,
    num_classes: int = 2,
) -> YOLO:
    """Build a YOLOv8 detection model for hold/volume detection.

    This function creates a YOLOv8 model configured for detecting
    climbing holds and volumes. By default, it uses the medium-sized
    YOLOv8m architecture with COCO pretrained weights.

    Args:
        model_size: YOLOv8 model variant to use. Options:
            - 'yolov8n': Nano (fastest, least accurate)
            - 'yolov8s': Small
            - 'yolov8m': Medium (default, good balance)
            - 'yolov8l': Large
            - 'yolov8x': Extra large (slowest, most accurate)
        pretrained: If True, load COCO pretrained weights.
            If False, initialize with random weights.
        num_classes: Number of output classes. Must be 2 for hold detection.

    Returns:
        Configured YOLO model ready for training or fine-tuning.

    Raises:
        ValueError: If num_classes is not 2 (hold, volume).
        ValueError: If model_size is not a valid YOLOv8 variant.

    Example:
        >>> model = build_hold_detector()
        >>> print(model.model.yaml['nc'])  # Number of classes
        2
        >>> print(model.names)
        {0: 'hold', 1: 'volume'}

        >>> # Build without pretrained weights
        >>> model = build_hold_detector(pretrained=False)

        >>> # Use larger model for better accuracy
        >>> model = build_hold_detector(model_size='yolov8l')
    """
    # Validate number of classes
    if num_classes != len(EXPECTED_CLASSES):
        raise ValueError(
            f"num_classes must be {len(EXPECTED_CLASSES)} for hold detection, "
            f"got {num_classes}"
        )

    # Validate model size
    if model_size not in VALID_MODEL_SIZES:
        raise ValueError(
            f"model_size must be one of {VALID_MODEL_SIZES}, got '{model_size}'"
        )

    logger.info(
        "Building %s detection model (pretrained=%s, num_classes=%d)",
        model_size,
        pretrained,
        num_classes,
    )

    # Build model path
    if pretrained:
        # Load pretrained model from Ultralytics
        model_path = f"{model_size}.pt"
    else:
        # Load architecture only (YAML config)
        model_path = f"{model_size}.yaml"

    # Create YOLO model
    model = YOLO(model_path)

    # Override number of classes if loading from YAML
    if not pretrained:
        # When loading from YAML, we can directly modify the model config
        model.model.yaml["nc"] = num_classes  # type: ignore[union-attr]
        model.model.yaml["names"] = EXPECTED_CLASSES.copy()  # type: ignore[union-attr]

    logger.info("Model created successfully: %s", model_path)
    logger.debug("Model configuration: nc=%d, names=%s", num_classes, EXPECTED_CLASSES)

    return model


def get_default_hyperparameters() -> DetectionHyperparameters:
    """Get default hyperparameters for hold detection training.

    Returns:
        DetectionHyperparameters instance with default values optimized
        for hold/volume detection.

    Example:
        >>> hyperparams = get_default_hyperparameters()
        >>> print(hyperparams.epochs)
        100
        >>> print(hyperparams.image_size)
        640
    """
    return DetectionHyperparameters()


def load_hyperparameters_from_file(config_path: Path | str) -> DetectionHyperparameters:
    """Load hyperparameters from a YAML configuration file.

    Args:
        config_path: Path to YAML file containing hyperparameter values.

    Returns:
        DetectionHyperparameters instance with values from file.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If YAML contains invalid hyperparameter values.

    Example:
        >>> hyperparams = load_hyperparameters_from_file("configs/training.yaml")
        >>> print(hyperparams.epochs)
        150
    """
    import yaml  # type: ignore[import-untyped]

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(
            f"Failed to parse YAML config file '{config_path}': {e}"
        ) from e

    if config_dict is None:
        config_dict = {}

    return DetectionHyperparameters(**config_dict)
