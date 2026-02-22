"""Hold classifier model definition for hold type classification.

This module provides functions to build and configure torchvision CNN models
(ResNet-18 or MobileNetV3) for classifying climbing hold types into the
6-class hold taxonomy used throughout the bouldering route analysis pipeline.

The final fully-connected layer of each backbone is replaced to match the
6-class output needed by CrossEntropyLoss.  No softmax is applied — raw
logits are returned so the training loop can use ``nn.CrossEntropyLoss``
directly.

Example:
    >>> from src.training.classification_model import build_hold_classifier
    >>> config = build_hold_classifier()
    >>> print(config["num_classes"])
    6
    >>> print(config["architecture"])
    resnet18
"""

from pathlib import Path
from typing import Any, TypedDict, cast

import yaml
from torch import nn
from torchvision import models
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.logging_config import get_logger
from src.training.classification_dataset import HOLD_CLASS_COUNT

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

DEFAULT_ARCHITECTURE: str = "resnet18"
INPUT_SIZE: int = 224  # Standard ImageNet / torchvision input resolution
VALID_ARCHITECTURES: tuple[str, ...] = (
    "resnet18",
    "mobilenet_v3_small",
    "mobilenet_v3_large",
)

# Final-layer in_features per backbone (private)
_RESNET18_FEATURES: int = 512
_MOBILENET_V3_SMALL_FEATURES: int = 1024
_MOBILENET_V3_LARGE_FEATURES: int = 1280


# ---------------------------------------------------------------------------
# Hyperparameter model
# ---------------------------------------------------------------------------


class ClassifierHyperparameters(BaseModel):
    """Hyperparameters for hold classifier model training.

    These parameters control the backbone selection, training schedule,
    optimizer, and regularisation settings for the ResNet-18 / MobileNetV3
    classification model.

    Attributes:
        architecture: CNN backbone to use. One of ``VALID_ARCHITECTURES``.
        pretrained: If True, initialise backbone with ImageNet weights.
        num_classes: Number of output classes (must match hold taxonomy).
        input_size: Square input resolution expected by the model.
        epochs: Number of training epochs.
        batch_size: Mini-batch size (must be ≥1).
        learning_rate: Initial learning rate (Adam-family optimizers).
        optimizer: Optimiser algorithm. One of ``{"SGD", "Adam", "AdamW"}``.
        weight_decay: L2 regularisation coefficient.
        momentum: Momentum for SGD.
        scheduler: LR scheduler. One of ``{"StepLR", "CosineAnnealingLR", "none"}``.
        label_smoothing: Label smoothing factor for CrossEntropyLoss [0, 1).
        dropout_rate: Dropout probability passed to the training loop.
            Not applied during model construction — the training loop (PR-4.4)
            is responsible for inserting ``nn.Dropout`` into the optimizer or
            loss pipeline.  Range [0, 1).
    """

    model_config = ConfigDict(populate_by_name=True)

    # Architecture
    architecture: str = Field(default=DEFAULT_ARCHITECTURE)
    pretrained: bool = Field(default=True)
    num_classes: int = Field(default=HOLD_CLASS_COUNT, ge=1, le=100)
    input_size: int = Field(default=INPUT_SIZE, ge=32, le=1024)

    # Training schedule
    epochs: int = Field(default=30, ge=1, le=1000)
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0.0, le=1.0)

    # Optimizer settings
    optimizer: str = Field(default="Adam")
    weight_decay: float = Field(default=1e-4, ge=0.0, le=1.0)
    momentum: float = Field(default=0.9, ge=0.0, le=1.0)

    # Scheduler
    scheduler: str = Field(default="StepLR")

    # Regularisation
    label_smoothing: float = Field(default=0.1, ge=0.0, lt=1.0)
    dropout_rate: float = Field(default=0.2, ge=0.0, lt=1.0)

    @field_validator("architecture")
    @classmethod
    def validate_architecture(cls, v: str) -> str:
        """Validate backbone architecture is supported.

        Args:
            v: Architecture name string.

        Returns:
            Validated architecture string.

        Raises:
            ValueError: If the architecture is not in VALID_ARCHITECTURES.
        """
        if v not in VALID_ARCHITECTURES:
            raise ValueError(
                f"architecture must be one of {VALID_ARCHITECTURES}, got '{v}'"
            )
        return v

    @field_validator("optimizer")
    @classmethod
    def validate_optimizer(cls, v: str) -> str:
        """Validate optimizer is supported.

        Args:
            v: Optimizer name string.

        Returns:
            Validated optimizer string.

        Raises:
            ValueError: If the optimizer is not in the supported set.
        """
        valid_optimizers = {"SGD", "Adam", "AdamW"}
        if v not in valid_optimizers:
            raise ValueError(f"optimizer must be one of {valid_optimizers}, got '{v}'")
        return v

    @field_validator("scheduler")
    @classmethod
    def validate_scheduler(cls, v: str) -> str:
        """Validate LR scheduler is supported.

        Args:
            v: Scheduler name string.

        Returns:
            Validated scheduler string.

        Raises:
            ValueError: If the scheduler is not in the supported set.
        """
        valid_schedulers = {"StepLR", "CosineAnnealingLR", "none"}
        if v not in valid_schedulers:
            raise ValueError(f"scheduler must be one of {valid_schedulers}, got '{v}'")
        return v

    def to_dict(self) -> dict[str, Any]:
        """Convert hyperparameters to a plain dictionary.

        Returns:
            Dictionary with field names (not aliases) and their values,
            excluding any None-valued fields.

        Example:
            >>> hp = ClassifierHyperparameters(epochs=50)
            >>> d = hp.to_dict()
            >>> print(d["epochs"])
            50
        """
        return self.model_dump(exclude_none=True)


# ---------------------------------------------------------------------------
# ClassifierConfig TypedDict
# ---------------------------------------------------------------------------


class ClassifierConfig(TypedDict):
    """Typed return value for :func:`build_hold_classifier`.

    Attributes:
        model: Configured ``nn.Module`` with final FC layer replaced.
        architecture: Backbone name (e.g. ``"resnet18"``).
        num_classes: Number of output classes (always 6 for hold taxonomy).
        input_size: Expected square input resolution (always 224).
        pretrained: Whether ImageNet weights were loaded.
    """

    model: nn.Module
    architecture: str
    num_classes: int
    input_size: int
    pretrained: bool


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_hold_classifier(
    hyperparameters: ClassifierHyperparameters | None = None,
) -> ClassifierConfig:
    """Build a torchvision CNN classifier for hold type classification.

    This function constructs a ResNet-18 or MobileNetV3 backbone (with optional
    ImageNet pre-training) and replaces the final fully-connected layer to
    produce logits over the 6-class hold taxonomy.

    No softmax is applied to the output — use ``nn.CrossEntropyLoss`` in the
    training loop, which expects raw logits.

    Args:
        hyperparameters: Configuration object controlling backbone selection,
            pretraining, and class count.  If ``None``, default values are
            used (ResNet-18, pretrained, 6 classes).

    Returns:
        :class:`ClassifierConfig` typed dict containing:
            - ``model``: Configured ``nn.Module`` ready for training.
            - ``architecture``: Backbone identifier string.
            - ``num_classes``: Number of output logits (6).
            - ``input_size``: Expected square input resolution (224).
            - ``pretrained``: Whether ImageNet weights were loaded.

    Raises:
        ValueError: If ``hyperparameters.num_classes`` does not equal
            ``HOLD_CLASS_COUNT`` (6).

    Example:
        >>> config = build_hold_classifier()
        >>> print(config["num_classes"])
        6

        >>> from src.training.classification_model import ClassifierHyperparameters
        >>> hp = ClassifierHyperparameters(architecture="mobilenet_v3_small", pretrained=False)
        >>> config = build_hold_classifier(hp)
        >>> print(config["architecture"])
        mobilenet_v3_small
    """
    if hyperparameters is None:
        hyperparameters = ClassifierHyperparameters()

    num_classes = hyperparameters.num_classes
    if num_classes != HOLD_CLASS_COUNT:
        raise ValueError(
            f"num_classes must be {HOLD_CLASS_COUNT} for hold classification, "
            f"got {num_classes}"
        )

    arch = hyperparameters.architecture
    pretrained = hyperparameters.pretrained
    input_size = hyperparameters.input_size

    logger.info(
        "Building %s classifier (pretrained=%s, num_classes=%d)",
        arch,
        pretrained,
        num_classes,
    )

    backbone = _build_backbone(arch, pretrained, num_classes)

    logger.info("Classifier built successfully: %s", arch)

    return ClassifierConfig(
        model=backbone,
        architecture=arch,
        num_classes=num_classes,
        input_size=input_size,
        pretrained=pretrained,
    )


def _build_backbone(arch: str, pretrained: bool, num_classes: int) -> nn.Module:
    """Construct and configure a torchvision backbone.

    Builds the requested backbone, optionally loading ImageNet weights, then
    replaces the final classification layer to match ``num_classes``.

    Args:
        arch: Backbone identifier.  Must be one of ``VALID_ARCHITECTURES``.
        pretrained: If True, load ImageNet weights (``weights=<Weights>.DEFAULT``).
        num_classes: Number of output logits for the final layer.

    Returns:
        Configured ``nn.Module`` with replaced final layer.

    Raises:
        ValueError: If ``arch`` is not in ``VALID_ARCHITECTURES``.
    """
    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(_RESNET18_FEATURES, num_classes)

    elif arch == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(_MOBILENET_V3_SMALL_FEATURES, num_classes)

    elif arch == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[-1] = nn.Linear(_MOBILENET_V3_LARGE_FEATURES, num_classes)

    else:
        raise ValueError(f"arch must be one of {VALID_ARCHITECTURES}, got '{arch}'")

    return cast(nn.Module, model)


def get_default_hyperparameters() -> ClassifierHyperparameters:
    """Get default hyperparameters for hold classification training.

    Returns:
        :class:`ClassifierHyperparameters` instance with default values:
        ResNet-18, pretrained, 30 epochs, Adam optimizer.

    Example:
        >>> hp = get_default_hyperparameters()
        >>> print(hp.architecture)
        resnet18
        >>> print(hp.epochs)
        30
    """
    return ClassifierHyperparameters()


def load_hyperparameters_from_file(
    config_path: Path | str,
) -> ClassifierHyperparameters:
    """Load classifier hyperparameters from a YAML configuration file.

    Args:
        config_path: Path to the YAML file.  Both ``Path`` objects and plain
            strings are accepted.

    Returns:
        :class:`ClassifierHyperparameters` instance with values from the file.
        Fields absent from the YAML retain their defaults.

    Raises:
        FileNotFoundError: If ``config_path`` does not exist.
        ValueError: If the YAML is syntactically invalid.
        pydantic.ValidationError: If the YAML contains invalid field values.

    Example:
        >>> hp = load_hyperparameters_from_file("configs/classifier.yaml")
        >>> print(hp.architecture)
        mobilenet_v3_small
    """
    path = Path(config_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise ValueError(
            f"Failed to parse YAML config file '{config_path}': {exc}"
        ) from exc

    if config_dict is None:
        config_dict = {}

    return ClassifierHyperparameters(**config_dict)
