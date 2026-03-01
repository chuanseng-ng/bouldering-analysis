"""Training loop for hold type classification models.

This module provides the main training orchestration for bouldering hold
classification.  It wraps PyTorch training with structured result models,
artifact saving, augmentation, and reproducibility metadata.

Supports ResNet-18 and MobileNetV3 backbones with:
- Weighted cross-entropy loss for class imbalance
- Adam / AdamW / SGD optimizers
- StepLR / CosineAnnealingLR schedulers
- Standard ImageNet augmentations (train) and centre-crop evaluation (val)
- Top-1 accuracy and Expected Calibration Error (ECE) metrics
- Versioned artifact layout under ``models/classification/<version>/``

Example:
    >>> from src.training.classification_dataset import load_hold_classification_dataset
    >>> from src.training.train_classification import train_hold_classifier
    >>> dataset = load_hold_classification_dataset("data/hold_classification")
    >>> result = train_hold_classifier(dataset, "data/hold_classification")
    >>> print(result.metrics.top1_accuracy)
    0.85
"""

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torchvision import transforms  # type: ignore[import-untyped]
from torchvision.datasets import ImageFolder  # type: ignore[import-untyped]

from src.logging_config import get_logger
from src.training.classification_dataset import ClassificationDatasetConfig
from src.training.classification_model import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    VALID_ARCHITECTURES,
    VAL_RESIZE_RATIO,
    ClassifierHyperparameters,
    apply_classifier_dropout,
    build_hold_classifier,
    get_default_hyperparameters,
)
from src.training.exceptions import ModelArtifactError, TrainingRunError

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS_BASE_DIR: Path = Path("models/classification")
METADATA_FILENAME: str = "metadata.json"
VERSION_FORMAT: str = "v%Y%m%d_%H%M%S"
# Windows requires num_workers=0 to avoid multiprocessing spawn issues with DataLoader
DEFAULT_NUM_WORKERS: int = 0 if platform.system() == "Windows" else 4


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class ClassificationMetrics(BaseModel):
    """Metrics captured at the end of a classification training run.

    Attributes:
        top1_accuracy: Top-1 accuracy on the validation split [0, 1].
        val_loss: Average cross-entropy loss on the validation split (≥0).
        ece: Expected Calibration Error on the validation split [0, 1].
        best_epoch: Epoch index at which the best checkpoint was saved.
    """

    top1_accuracy: float = Field(ge=0.0, le=1.0)
    val_loss: float = Field(ge=0.0)
    ece: float = Field(ge=0.0, le=1.0)
    best_epoch: int = Field(ge=0)


class ClassificationTrainingResult(BaseModel):
    """Full output of a completed classification training run.

    Attributes:
        version: Unique version string in the format v%Y%m%d_%H%M%S.
        architecture: CNN backbone used (e.g. 'resnet18').
        best_weights_path: Absolute path to best.pt checkpoint.
        last_weights_path: Absolute path to last.pt checkpoint.
        metadata_path: Absolute path to metadata.json artifact.
        metrics: Training metrics from the best epoch.
        dataset_root: Path to the classification dataset used for training.
        git_commit: Short git commit hash at training time, or None.
        trained_at: ISO-8601 UTC timestamp when training completed.
        hyperparameters: Dictionary of hyperparameter values used.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    version: str
    architecture: str
    best_weights_path: Path
    last_weights_path: Path
    metadata_path: Path
    metrics: ClassificationMetrics
    dataset_root: str
    git_commit: str | None
    trained_at: str
    hyperparameters: dict[str, Any]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _generate_version() -> str:
    """Generate a unique version string based on the current UTC time.

    Returns:
        Version string in the format v%Y%m%d_%H%M%S (e.g. 'v20260222_120000').
    """
    return datetime.now(tz=timezone.utc).strftime(VERSION_FORMAT)


def _get_git_commit_hash() -> str | None:
    """Get the short git commit hash of the current HEAD.

    Returns:
        Short commit hash string (e.g. 'abc1234'), or None if git is
        unavailable or the command fails.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None
    except subprocess.TimeoutExpired:
        logger.debug("git rev-parse timed out; git_commit will be None")
        return None
    except Exception as exc:  # noqa: BLE001  # pylint: disable=broad-exception-caught
        logger.debug("Could not get git commit hash: %s", exc, exc_info=True)
        return None


def _get_device() -> torch.device:
    """Return the best available compute device.

    Returns:
        ``torch.device("cuda")`` if a CUDA-capable GPU is available,
        otherwise ``torch.device("cpu")``.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _apply_dropout(model: nn.Module, hp: ClassifierHyperparameters) -> nn.Module:
    """Return a new model with Dropout inserted before the final classifier layer.

    PR-4.3 deliberately omits dropout from the model builder so that the
    training loop can insert it here.  A deep copy of the model is made first
    so the original object is never mutated (per the project immutability rule).

    The wrapping pattern is ``nn.Sequential(nn.Dropout(p), original_layer)``.

    Returns the original model unchanged when ``hp.dropout_rate <= 0``.

    - **ResNet-18**: wraps ``model.fc``
    - **MobileNetV3 (small/large)**: wraps ``model.classifier[-1]``

    Args:
        model: The ``nn.Module`` returned by ``build_hold_classifier()``.
            Not mutated — a deep copy is created and returned.
        hp: Hyperparameters containing ``architecture`` and ``dropout_rate``.

    Returns:
        A new ``nn.Module`` with the final layer wrapped in Dropout, or the
        original model if ``hp.dropout_rate <= 0``.

    Raises:
        TrainingRunError: If ``hp.architecture`` is not a recognised value.
    """
    try:
        return apply_classifier_dropout(model, hp.architecture, hp.dropout_rate)
    except ValueError as exc:
        raise TrainingRunError(
            f"Unsupported architecture for dropout insertion: '{hp.architecture}'. "
            f"Must be one of {VALID_ARCHITECTURES}."
        ) from exc


def _build_transforms(
    hp: ClassifierHyperparameters,
    training: bool,
) -> transforms.Compose:
    """Build torchvision transform pipelines.

    Training pipeline applies data augmentation (random crops, flips,
    rotation, colour jitter, random erasing) to improve generalisation.

    Validation pipeline applies deterministic centre-crop evaluation.

    Both pipelines normalise with standard ImageNet statistics.

    Args:
        hp: Hyperparameters providing ``input_size``.
        training: If True, return augmented training transforms; otherwise
            return deterministic validation transforms.

    Returns:
        ``transforms.Compose`` pipeline ready to pass to ``ImageFolder``.
    """
    size = hp.input_size

    if training:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.3,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.1,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=list(IMAGENET_MEAN),
                    std=list(IMAGENET_STD),
                ),
                transforms.RandomErasing(p=0.2),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(int(size * VAL_RESIZE_RATIO)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=list(IMAGENET_MEAN),
                std=list(IMAGENET_STD),
            ),
        ]
    )


def _build_data_loaders(
    dataset: ClassificationDatasetConfig,
    hp: ClassifierHyperparameters,
) -> tuple[DataLoader, DataLoader]:  # type: ignore[type-arg]
    """Build training and validation DataLoaders from an ImageFolder dataset.

    Args:
        dataset: Validated classification dataset config with ``train`` and
            ``val`` split paths.
        hp: Hyperparameters controlling ``batch_size``.

    Returns:
        Tuple of ``(train_loader, val_loader)`` DataLoaders.
    """
    train_ds = ImageFolder(
        root=str(dataset["train"]),
        transform=_build_transforms(hp, training=True),
    )
    val_ds = ImageFolder(
        root=str(dataset["val"]),
        transform=_build_transforms(hp, training=False),
    )

    train_loader: DataLoader = DataLoader(  # type: ignore[type-arg]
        train_ds,
        batch_size=hp.batch_size,
        shuffle=True,
        num_workers=DEFAULT_NUM_WORKERS,
        pin_memory=True,
    )
    val_loader: DataLoader = DataLoader(  # type: ignore[type-arg]
        val_ds,
        batch_size=hp.batch_size,
        shuffle=False,
        num_workers=DEFAULT_NUM_WORKERS,
        pin_memory=True,
    )
    return train_loader, val_loader


def _build_optimizer(
    model: nn.Module,
    hp: ClassifierHyperparameters,
) -> Optimizer:
    """Construct the gradient-descent optimizer.

    Supports Adam, AdamW, and SGD.  SGD uses ``hp.momentum``; Adam and
    AdamW use ``hp.weight_decay``.

    Args:
        model: The model whose parameters will be optimised.
        hp: Hyperparameters containing ``optimizer``, ``learning_rate``,
            ``weight_decay``, and ``momentum``.

    Returns:
        A configured ``torch.optim.Optimizer``.

    Raises:
        TrainingRunError: If ``hp.optimizer`` is not a supported value.
    """
    if hp.optimizer == "Adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=hp.learning_rate,
            weight_decay=hp.weight_decay,
        )
    if hp.optimizer == "AdamW":
        return torch.optim.AdamW(
            model.parameters(),
            lr=hp.learning_rate,
            weight_decay=hp.weight_decay,
        )
    if hp.optimizer == "SGD":
        return torch.optim.SGD(
            model.parameters(),
            lr=hp.learning_rate,
            momentum=hp.momentum,
            weight_decay=hp.weight_decay,
        )
    raise TrainingRunError(
        f"Unsupported optimizer: '{hp.optimizer}'. "
        "Must be one of 'Adam', 'AdamW', 'SGD'."
    )


def _build_scheduler(
    optimizer: Optimizer,
    hp: ClassifierHyperparameters,
) -> LRScheduler | None:
    """Construct the learning-rate scheduler.

    Supports StepLR (halves LR every ``epochs//3`` epochs), CosineAnnealingLR, and
    ``'none'`` (no scheduling).

    Args:
        optimizer: The optimizer whose learning rate will be scheduled.
        hp: Hyperparameters containing ``scheduler`` and ``epochs``.

    Returns:
        A configured ``torch.optim.lr_scheduler.LRScheduler``, or ``None``
        when ``hp.scheduler == 'none'``.
    """
    if hp.scheduler == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, hp.epochs // 3),
            gamma=0.5,
        )
    if hp.scheduler == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=hp.epochs,
        )
    # hp.scheduler == "none"
    return None


def _run_train_epoch(
    model: nn.Module,
    loader: DataLoader,  # type: ignore[type-arg]
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch and return average loss and Top-1 accuracy.

    Sets the model to training mode, iterates over all batches, computes
    loss and backprop, and calls ``optimizer.step()``.

    Args:
        model: Classification model to train.
        loader: Training DataLoader.
        criterion: Loss function (e.g. ``nn.CrossEntropyLoss``).
        optimizer: Gradient-descent optimizer.
        device: Compute device for tensor operations.

    Returns:
        Tuple of ``(avg_loss, top1_accuracy)`` as Python floats.
    """
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

    if total_samples == 0:
        return 0.0, 0.0

    return total_loss / total_samples, total_correct / total_samples


def _run_val_epoch(  # pylint: disable=too-many-locals
    model: nn.Module,
    loader: DataLoader,  # type: ignore[type-arg]
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, torch.Tensor, torch.Tensor]:
    """Run one validation epoch and return loss, accuracy, probs, and labels.

    Sets the model to eval mode and disables gradient computation.

    Args:
        model: Classification model to evaluate.
        loader: Validation DataLoader.
        criterion: Loss function.
        device: Compute device.

    Returns:
        Tuple of ``(avg_loss, top1_accuracy, all_probs, all_labels)`` where
        ``all_probs`` has shape ``(N, num_classes)`` and ``all_labels`` has
        shape ``(N,)``.
    """
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_probs_list: list[torch.Tensor] = []
    all_labels_list: list[torch.Tensor] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (preds == labels).sum().item()
            total_samples += batch_size

            all_probs_list.append(probs.cpu())
            all_labels_list.append(labels.cpu())

    if total_samples == 0:
        weight: torch.Tensor | None = getattr(criterion, "weight", None)
        num_classes: int = int(weight.shape[0]) if weight is not None else 1
        empty = torch.zeros(0, num_classes)
        return 0.0, 0.0, empty, torch.zeros(0, dtype=torch.long)

    all_probs = torch.cat(all_probs_list, dim=0)
    all_labels = torch.cat(all_labels_list, dim=0)
    return (
        total_loss / total_samples,
        total_correct / total_samples,
        all_probs,
        all_labels,
    )


def _compute_ece(  # pylint: disable=too-many-locals
    probs: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """Compute the Expected Calibration Error (ECE).

    Measures the gap between a model's confidence and its empirical accuracy
    by binning predictions into ``n_bins`` equal-width confidence intervals
    and computing a bin-size-weighted average of ``|confidence - accuracy|``.

    Args:
        probs: Predicted class probabilities, shape ``(N, num_classes)``.
            Each row must sum to 1.
        labels: Ground-truth class indices, shape ``(N,)``.
        n_bins: Number of equally-spaced confidence bins (default: 15).

    Returns:
        ECE as a Python float in ``[0, 1]``.
    """
    confidences, preds = probs.max(dim=1)
    correct = preds.eq(labels)

    n = len(labels)
    if n == 0:
        return 0.0

    ece = 0.0
    bin_boundaries = torch.linspace(0.0, 1.0, n_bins + 1)

    for i in range(n_bins):
        lower = bin_boundaries[i].item()
        upper = bin_boundaries[i + 1].item()
        # Include upper bound only for the last bin
        if i < n_bins - 1:
            mask = (confidences >= lower) & (confidences < upper)
        else:
            mask = (confidences >= lower) & (confidences <= upper)

        if mask.sum() == 0:
            continue

        bin_conf = confidences[mask].mean().item()
        bin_acc = correct[mask].float().mean().item()
        bin_weight = mask.sum().item() / n
        ece += bin_weight * abs(bin_conf - bin_acc)

    return float(ece)


def _build_metadata(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    version: str,
    architecture: str,
    trained_at: str,
    git_commit: str | None,
    dataset_root: str,
    hyperparameters: ClassifierHyperparameters,
    metrics: ClassificationMetrics,
) -> dict[str, Any]:
    """Assemble the metadata dictionary to be serialised as metadata.json.

    Args:
        version: Run version string (e.g. 'v20260222_120000').
        architecture: CNN backbone name (e.g. 'resnet18').
        trained_at: ISO-8601 UTC timestamp string.
        git_commit: Short git commit hash or None.
        dataset_root: Path string to the dataset root.
        hyperparameters: Hyperparameters used for training.
        metrics: Final training metrics.

    Returns:
        JSON-serialisable dictionary with all required metadata fields.
    """
    return {
        "version": version,
        "architecture": architecture,
        "trained_at": trained_at,
        "git_commit": git_commit,
        "dataset_root": dataset_root,
        "hyperparameters": hyperparameters.to_dict(),
        "metrics": metrics.model_dump(),
    }


def _save_artifacts(
    best_state: dict[str, Any],
    last_state: dict[str, Any],
    metadata: dict[str, Any],
    out_dir: Path,
    version: str,
) -> tuple[Path, Path, Path]:
    """Save training artifacts to a versioned output directory.

    Creates the following layout::

        <out_dir>/<version>/
            weights/
                best.pt
                last.pt
            metadata.json

    Args:
        best_state: ``state_dict`` of the best-val-loss model.
        last_state: ``state_dict`` of the model at the final epoch.
        metadata: Dictionary to serialise as metadata.json.
        out_dir: Base output directory for all model versions.
        version: Version string (used as sub-directory name).

    Returns:
        Tuple of ``(best_weights_path, last_weights_path, metadata_path)``.

    Raises:
        ModelArtifactError: If the output directory cannot be created or
            weight files cannot be written.
    """
    version_dir = out_dir / version
    weights_dir = version_dir / "weights"

    try:
        weights_dir.mkdir(parents=True, exist_ok=True)

        best_path = weights_dir / "best.pt"
        last_path = weights_dir / "last.pt"
        torch.save(best_state, best_path)
        torch.save(last_state, last_path)

        meta_path = version_dir / METADATA_FILENAME
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return best_path, last_path, meta_path

    except OSError as exc:
        raise ModelArtifactError(
            f"Failed to save classification training artifacts: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train_hold_classifier(  # pylint: disable=too-many-locals
    dataset: ClassificationDatasetConfig,
    dataset_root: Path | str,
    hyperparameters: ClassifierHyperparameters | None = None,
    output_dir: Path | str | None = None,
) -> ClassificationTrainingResult:
    """Train a ResNet-18 or MobileNetV3 hold classifier.

    Orchestrates the full classification training pipeline:

    1. Resolve hyperparameters and output directory.
    2. Build the model via :func:`build_hold_classifier` and apply dropout.
    3. Build data loaders, optimizer, scheduler, and loss function.
    4. Run ``hp.epochs`` training + validation epochs.
    5. Track best checkpoint by minimum validation loss.
    6. Compute ECE on the final best validation predictions.
    7. Save artifacts (``best.pt``, ``last.pt``, ``metadata.json``) to
       a versioned sub-directory.

    Args:
        dataset: Validated :class:`ClassificationDatasetConfig` from
            :func:`~src.training.classification_dataset.load_hold_classification_dataset`.
            Provides split paths and class weights.
        dataset_root: Path to the dataset root directory (stored in metadata
            for reproducibility).
        hyperparameters: Training hyperparameters.  Uses defaults if ``None``.
        output_dir: Base directory to save model artifacts.  Defaults to
            ``MODELS_BASE_DIR`` (``models/classification``).

    Returns:
        :class:`ClassificationTrainingResult` with paths to saved artifacts
        and training metrics.

    Raises:
        TrainingRunError: If an unsupported optimizer is specified or the
            training loop encounters a fatal error.
        ModelArtifactError: If saving artifacts fails after training.

    Example:
        >>> from src.training.classification_dataset import load_hold_classification_dataset
        >>> from src.training.train_classification import train_hold_classifier
        >>> dataset = load_hold_classification_dataset("data/hold_classification")
        >>> result = train_hold_classifier(dataset, "data/hold_classification")
        >>> print(result.metrics.top1_accuracy)
        0.85
    """
    if hyperparameters is None:
        hyperparameters = get_default_hyperparameters()

    resolved_root = str(Path(dataset_root).resolve())
    resolved_output = (
        Path(output_dir).resolve()
        if output_dir is not None
        else MODELS_BASE_DIR.resolve()
    )

    # Capture a single timestamp so version and trained_at match exactly
    now = datetime.now(tz=timezone.utc)
    version = now.strftime(VERSION_FORMAT)
    trained_at = now.isoformat()
    git_commit = _get_git_commit_hash()

    logger.info(
        "Starting classification training run %s (arch=%s, epochs=%d)",
        version,
        hyperparameters.architecture,
        hyperparameters.epochs,
    )

    device = _get_device()

    # Build model and apply dropout
    config = build_hold_classifier(hyperparameters)
    model: nn.Module = config["model"]
    model = _apply_dropout(model, hyperparameters)
    model = model.to(device)

    # Build data pipeline
    train_loader, val_loader = _build_data_loaders(dataset, hyperparameters)

    # Build class-weighted loss
    class_weights_tensor = torch.tensor(
        dataset["class_weights"], dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor,
        label_smoothing=hyperparameters.label_smoothing,
    )

    optimizer = _build_optimizer(model, hyperparameters)
    scheduler = _build_scheduler(optimizer, hyperparameters)

    # Training loop
    best_val_loss = float("inf")
    best_state: dict[str, Any] = {k: v.clone() for k, v in model.state_dict().items()}
    last_state: dict[str, Any] = {}
    best_epoch = 0
    best_top1 = 0.0
    best_probs = torch.zeros(0, hyperparameters.num_classes)
    best_labels = torch.zeros(0, dtype=torch.long)

    for epoch in range(hyperparameters.epochs):
        train_loss, train_acc = _run_train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_probs, val_labels = _run_val_epoch(
            model, val_loader, criterion, device
        )

        if scheduler is not None:
            scheduler.step()

        logger.info(
            "Epoch %d/%d — train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
            epoch + 1,
            hyperparameters.epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_top1 = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_probs = val_probs.clone()
            best_labels = val_labels.clone()

    # Capture the final epoch state once, outside the loop, to avoid
    # redundant O(params) clones on every epoch.
    last_state = {k: v.clone() for k, v in model.state_dict().items()}

    ece = _compute_ece(best_probs, best_labels)
    metrics = ClassificationMetrics(
        top1_accuracy=best_top1,
        val_loss=best_val_loss if best_val_loss != float("inf") else 0.0,
        ece=ece,
        best_epoch=best_epoch,
    )

    metadata = _build_metadata(
        version=version,
        architecture=hyperparameters.architecture,
        trained_at=trained_at,
        git_commit=git_commit,
        dataset_root=resolved_root,
        hyperparameters=hyperparameters,
        metrics=metrics,
    )

    best_path, last_path, meta_path = _save_artifacts(
        best_state=best_state,
        last_state=last_state,
        metadata=metadata,
        out_dir=resolved_output,
        version=version,
    )

    logger.info(
        "Classification training complete. top1=%.3f ECE=%.3f best_epoch=%d artifacts=%s",
        metrics.top1_accuracy,
        metrics.ece,
        metrics.best_epoch,
        best_path.parent.parent,
    )

    return ClassificationTrainingResult(
        version=version,
        architecture=hyperparameters.architecture,
        best_weights_path=best_path,
        last_weights_path=last_path,
        metadata_path=meta_path,
        metrics=metrics,
        dataset_root=resolved_root,
        git_commit=git_commit,
        trained_at=trained_at,
        hyperparameters=hyperparameters.to_dict(),
    )
