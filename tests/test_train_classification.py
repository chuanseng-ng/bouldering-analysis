"""Tests for the hold classification training loop.

This module tests the training pipeline for classification models,
including result models, helpers, and the main train_hold_classifier
function.

Tests follow TDD: written before implementation.
"""

# pylint: disable=redefined-outer-name  # standard pytest fixture pattern
# pylint: disable=too-many-lines  # comprehensive test coverage requires many test cases

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from pydantic import ValidationError
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from torchvision import models  # type: ignore[import-untyped]
from torchvision.transforms import Compose, Normalize  # type: ignore[import-untyped]

from src.training.classification_dataset import ClassificationDatasetConfig
from src.training.classification_model import ClassifierHyperparameters
from src.training.exceptions import ModelArtifactError, TrainingRunError
from src.training.train_classification import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ClassificationMetrics,
    ClassificationTrainingResult,
    _apply_dropout,
    _build_data_loaders,
    _build_metadata,
    _build_optimizer,
    _build_scheduler,
    _build_transforms,
    _compute_ece,
    _generate_version,
    _get_device,
    _get_git_commit_hash,
    _run_train_epoch,
    _run_val_epoch,
    _save_artifacts,
    train_hold_classifier,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_classification_dataset_config(
    train: Path,
    val: Path,
) -> ClassificationDatasetConfig:
    """Create a minimal ClassificationDatasetConfig for testing."""
    return ClassificationDatasetConfig(
        train=train,
        val=val,
        test=None,
        nc=6,
        names=["jug", "crimp", "sloper", "pinch", "volume", "unknown"],
        train_image_count=60,
        val_image_count=12,
        test_image_count=0,
        class_counts={
            "jug": 10,
            "crimp": 10,
            "sloper": 10,
            "pinch": 10,
            "volume": 10,
            "unknown": 10,
        },
        class_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        version=None,
        metadata={},
    )


@pytest.fixture
def tmp_dataset(tmp_path: Path) -> ClassificationDatasetConfig:
    """Fixture providing a minimal valid classification dataset structure."""
    hold_classes = ("jug", "crimp", "sloper", "pinch", "volume", "unknown")
    for split in ("train", "val"):
        for cls in hold_classes:
            cls_dir = tmp_path / split / cls
            cls_dir.mkdir(parents=True)
            # Create a tiny valid RGB PNG (1x1 pixel)
            try:
                from PIL import Image  # type: ignore[import-untyped]  # pylint: disable=import-outside-toplevel

                img = Image.new("RGB", (64, 64), color=(128, 64, 32))
                img.save(cls_dir / "sample.jpg")
            except ImportError:
                # Fallback: minimal valid JPEG header
                (cls_dir / "sample.jpg").write_bytes(
                    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
                    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
                    b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
                    b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e\xff\xd9"
                )

    return _make_classification_dataset_config(
        train=tmp_path / "train",
        val=tmp_path / "val",
    )


@pytest.fixture
def minimal_resnet() -> nn.Module:
    """Fixture providing a tiny ResNet-18 model for unit tests."""
    model = models.resnet18(weights=None)
    # Replace final layer to match 6 classes
    model.fc = nn.Linear(model.fc.in_features, 6)
    return model  # type: ignore[return-value,no-any-return]


@pytest.fixture
def minimal_mobilenet() -> nn.Module:
    """Fixture providing a tiny MobileNetV3-Small model for unit tests."""
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(  # type: ignore[index]
        model.classifier[-1].in_features,  # type: ignore[union-attr]
        6,
    )
    return model  # type: ignore[return-value,no-any-return]


# ---------------------------------------------------------------------------
# TestClassificationMetrics
# ---------------------------------------------------------------------------


class TestClassificationMetrics:
    """Tests for ClassificationMetrics Pydantic model."""

    def test_valid_construction(self) -> None:
        """ClassificationMetrics accepts all-valid fields."""
        m = ClassificationMetrics(
            top1_accuracy=0.82,
            val_loss=0.45,
            ece=0.03,
            best_epoch=12,
        )
        assert m.top1_accuracy == pytest.approx(0.82)
        assert m.val_loss == pytest.approx(0.45)
        assert m.ece == pytest.approx(0.03)
        assert m.best_epoch == 12

    def test_accuracy_bounds_zero(self) -> None:
        """ClassificationMetrics accepts 0.0 for accuracy."""
        m = ClassificationMetrics(
            top1_accuracy=0.0, val_loss=0.0, ece=0.0, best_epoch=0
        )
        assert m.top1_accuracy == 0.0

    def test_accuracy_bounds_one(self) -> None:
        """ClassificationMetrics accepts 1.0 for accuracy."""
        m = ClassificationMetrics(
            top1_accuracy=1.0, val_loss=0.1, ece=1.0, best_epoch=99
        )
        assert m.top1_accuracy == 1.0

    def test_rejects_accuracy_above_one(self) -> None:
        """ClassificationMetrics rejects accuracy > 1.0."""
        with pytest.raises(ValidationError):
            ClassificationMetrics(
                top1_accuracy=1.1, val_loss=0.1, ece=0.05, best_epoch=0
            )

    def test_rejects_accuracy_below_zero(self) -> None:
        """ClassificationMetrics rejects accuracy < 0.0."""
        with pytest.raises(ValidationError):
            ClassificationMetrics(
                top1_accuracy=-0.1, val_loss=0.1, ece=0.05, best_epoch=0
            )

    def test_rejects_negative_val_loss(self) -> None:
        """ClassificationMetrics rejects val_loss < 0.0."""
        with pytest.raises(ValidationError):
            ClassificationMetrics(
                top1_accuracy=0.5, val_loss=-0.1, ece=0.05, best_epoch=0
            )

    def test_rejects_ece_above_one(self) -> None:
        """ClassificationMetrics rejects ECE > 1.0."""
        with pytest.raises(ValidationError):
            ClassificationMetrics(
                top1_accuracy=0.5, val_loss=0.1, ece=1.1, best_epoch=0
            )

    def test_rejects_negative_epoch(self) -> None:
        """ClassificationMetrics rejects best_epoch < 0."""
        with pytest.raises(ValidationError):
            ClassificationMetrics(
                top1_accuracy=0.5, val_loss=0.1, ece=0.05, best_epoch=-1
            )


# ---------------------------------------------------------------------------
# TestClassificationTrainingResult
# ---------------------------------------------------------------------------


class TestClassificationTrainingResult:
    """Tests for ClassificationTrainingResult Pydantic model."""

    def test_valid_construction(self, tmp_path: Path) -> None:
        """ClassificationTrainingResult can be constructed with all fields."""
        metrics = ClassificationMetrics(
            top1_accuracy=0.85, val_loss=0.33, ece=0.04, best_epoch=15
        )
        result = ClassificationTrainingResult(
            version="v20260222_120000",
            architecture="resnet18",
            best_weights_path=tmp_path / "best.pt",
            last_weights_path=tmp_path / "last.pt",
            metadata_path=tmp_path / "metadata.json",
            metrics=metrics,
            dataset_root="data/hold_classification",
            git_commit="abc1234",
            trained_at="2026-02-22T12:00:00Z",
            hyperparameters={"epochs": 30, "batch_size": 32},
        )
        assert result.version == "v20260222_120000"
        assert result.architecture == "resnet18"
        assert result.git_commit == "abc1234"
        assert isinstance(result.best_weights_path, Path)

    def test_optional_git_commit_none(self, tmp_path: Path) -> None:
        """ClassificationTrainingResult allows git_commit to be None."""
        metrics = ClassificationMetrics(
            top1_accuracy=0.5, val_loss=1.0, ece=0.1, best_epoch=0
        )
        result = ClassificationTrainingResult(
            version="v20260222_000000",
            architecture="mobilenet_v3_small",
            best_weights_path=tmp_path / "best.pt",
            last_weights_path=tmp_path / "last.pt",
            metadata_path=tmp_path / "metadata.json",
            metrics=metrics,
            dataset_root="data/hold_classification",
            git_commit=None,
            trained_at="2026-02-22T00:00:00Z",
            hyperparameters={},
        )
        assert result.git_commit is None

    def test_path_fields_are_path_objects(self, tmp_path: Path) -> None:
        """ClassificationTrainingResult stores Path objects for path fields."""
        metrics = ClassificationMetrics(
            top1_accuracy=0.5, val_loss=1.0, ece=0.1, best_epoch=0
        )
        result = ClassificationTrainingResult(
            version="v20260222_000000",
            architecture="resnet18",
            best_weights_path=tmp_path / "best.pt",
            last_weights_path=tmp_path / "last.pt",
            metadata_path=tmp_path / "metadata.json",
            metrics=metrics,
            dataset_root="data/hold_classification",
            git_commit=None,
            trained_at="2026-02-22T00:00:00Z",
            hyperparameters={},
        )
        assert isinstance(result.best_weights_path, Path)
        assert isinstance(result.last_weights_path, Path)
        assert isinstance(result.metadata_path, Path)


# ---------------------------------------------------------------------------
# TestGenerateVersion
# ---------------------------------------------------------------------------


class TestGenerateVersion:
    """Tests for _generate_version helper."""

    def test_version_format(self) -> None:
        """_generate_version returns string matching v\\d{8}_\\d{6}."""
        version = _generate_version()
        assert re.match(r"^v\d{8}_\d{6}$", version), f"Bad format: {version!r}"

    def test_version_starts_with_v(self) -> None:
        """_generate_version always starts with 'v'."""
        assert _generate_version().startswith("v")

    def test_versions_unique_across_seconds(self) -> None:
        """Two calls with different timestamps produce different versions."""
        t1 = datetime(2026, 2, 22, 12, 0, 0, tzinfo=timezone.utc)
        t2 = datetime(2026, 2, 22, 12, 0, 1, tzinfo=timezone.utc)
        with patch("src.training.train_classification.datetime") as mock_dt:
            mock_dt.now.return_value = t1
            v1 = _generate_version()
            mock_dt.now.return_value = t2
            v2 = _generate_version()
        assert v1 != v2


# ---------------------------------------------------------------------------
# TestGetGitCommitHash
# ---------------------------------------------------------------------------


class TestGetGitCommitHash:
    """Tests for _get_git_commit_hash helper."""

    def test_returns_none_on_file_not_found(self) -> None:
        """_get_git_commit_hash returns None when git binary not found."""
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = _get_git_commit_hash()
        assert result is None

    def test_returns_none_on_nonzero_returncode(self) -> None:
        """_get_git_commit_hash returns None when git command fails."""
        mock_result = MagicMock()
        mock_result.returncode = 128
        mock_result.stdout = ""
        with patch("subprocess.run", return_value=mock_result):
            result = _get_git_commit_hash()
        assert result is None

    def test_returns_string_on_success(self) -> None:
        """_get_git_commit_hash returns hex string on success."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "abc1234\n"
        with patch("subprocess.run", return_value=mock_result):
            result = _get_git_commit_hash()
        assert result == "abc1234"

    def test_returns_none_on_empty_stdout(self) -> None:
        """_get_git_commit_hash returns None when stdout is empty."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        with patch("subprocess.run", return_value=mock_result):
            result = _get_git_commit_hash()
        assert result is None


# ---------------------------------------------------------------------------
# TestGetDevice
# ---------------------------------------------------------------------------


class TestGetDevice:
    """Tests for _get_device helper."""

    def test_returns_torch_device(self) -> None:
        """_get_device returns a torch.device instance."""
        device = _get_device()
        assert isinstance(device, torch.device)

    def test_device_is_cpu_or_cuda(self) -> None:
        """_get_device returns cpu or cuda device."""
        device = _get_device()
        assert device.type in ("cpu", "cuda")


# ---------------------------------------------------------------------------
# TestApplyDropout
# ---------------------------------------------------------------------------


class TestApplyDropout:  # pylint: disable=too-many-public-methods
    """Tests for _apply_dropout helper."""

    def test_resnet_fc_wrapped_with_dropout(self, minimal_resnet: nn.Module) -> None:
        """_apply_dropout wraps ResNet model.fc in nn.Sequential with Dropout."""
        hp = ClassifierHyperparameters(architecture="resnet18", dropout_rate=0.3)
        model = _apply_dropout(minimal_resnet, hp)
        # fc should now be Sequential
        fc = model.fc  # type: ignore[union-attr]
        assert isinstance(fc, nn.Sequential)
        # First element should be Dropout
        assert isinstance(fc[0], nn.Dropout)
        assert fc[0].p == pytest.approx(0.3)

    def test_mobilenet_classifier_last_wrapped(
        self, minimal_mobilenet: nn.Module
    ) -> None:
        """_apply_dropout wraps MobileNet classifier[-1] with Dropout."""
        hp = ClassifierHyperparameters(
            architecture="mobilenet_v3_small", dropout_rate=0.25
        )
        model = _apply_dropout(minimal_mobilenet, hp)
        last_layer = model.classifier[-1]  # type: ignore[index]
        assert isinstance(last_layer, nn.Sequential)
        assert isinstance(last_layer[0], nn.Dropout)
        assert last_layer[0].p == pytest.approx(0.25)

    def test_zero_dropout_rate_skips_wrap_resnet(
        self, minimal_resnet: nn.Module
    ) -> None:
        """_apply_dropout skips wrapping when dropout_rate == 0.0."""
        hp = ClassifierHyperparameters(architecture="resnet18", dropout_rate=0.0)
        original_fc = minimal_resnet.fc  # type: ignore[union-attr]
        model = _apply_dropout(minimal_resnet, hp)
        # Should remain as Linear, not Sequential
        assert isinstance(model.fc, nn.Linear)  # type: ignore[union-attr]
        assert model.fc is original_fc  # type: ignore[union-attr]

    def test_zero_dropout_rate_skips_wrap_mobilenet(
        self, minimal_mobilenet: nn.Module
    ) -> None:
        """_apply_dropout skips wrapping mobilenet when dropout_rate == 0.0."""
        hp = ClassifierHyperparameters(
            architecture="mobilenet_v3_small", dropout_rate=0.0
        )
        original_last = minimal_mobilenet.classifier[-1]  # type: ignore[index]
        model = _apply_dropout(minimal_mobilenet, hp)
        assert model.classifier[-1] is original_last  # type: ignore[index]

    def test_mobilenet_v3_large_classifier_last_wrapped(self) -> None:
        """_apply_dropout wraps mobilenet_v3_large classifier[-1] with Dropout."""
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[-1] = nn.Linear(  # type: ignore[index,union-attr]
            model.classifier[-1].in_features,
            6,  # type: ignore[union-attr]
        )
        hp = ClassifierHyperparameters(
            architecture="mobilenet_v3_large", dropout_rate=0.1
        )
        result = _apply_dropout(model, hp)  # type: ignore[arg-type]
        last_layer = result.classifier[-1]  # type: ignore[index]
        assert isinstance(last_layer, nn.Sequential)
        assert isinstance(last_layer[0], nn.Dropout)

    def test_unsupported_architecture_raises_training_run_error(
        self, minimal_resnet: nn.Module
    ) -> None:
        """_apply_dropout raises TrainingRunError for unknown architecture."""
        hp = ClassifierHyperparameters(architecture="resnet18", dropout_rate=0.2)
        object.__setattr__(hp, "architecture", "efficientnet_b0")
        with pytest.raises(TrainingRunError):
            _apply_dropout(minimal_resnet, hp)

    def test_returns_nn_module(self, minimal_resnet: nn.Module) -> None:
        """_apply_dropout returns nn.Module."""
        hp = ClassifierHyperparameters(architecture="resnet18", dropout_rate=0.2)
        result = _apply_dropout(minimal_resnet, hp)
        assert isinstance(result, nn.Module)


# ---------------------------------------------------------------------------
# TestBuildTransforms
# ---------------------------------------------------------------------------


class TestBuildTransforms:
    """Tests for _build_transforms helper."""

    def test_training_transforms_include_random_rotation(self) -> None:
        """Training transforms include RandomRotation."""
        hp = ClassifierHyperparameters()
        t = _build_transforms(hp, training=True)
        assert isinstance(t, Compose)
        transform_names = [type(tr).__name__ for tr in t.transforms]
        assert "RandomRotation" in transform_names

    def test_training_transforms_include_color_jitter(self) -> None:
        """Training transforms include ColorJitter."""
        hp = ClassifierHyperparameters()
        t = _build_transforms(hp, training=True)
        transform_names = [type(tr).__name__ for tr in t.transforms]
        assert "ColorJitter" in transform_names

    def test_training_transforms_include_random_erasing(self) -> None:
        """Training transforms include RandomErasing (cutout augmentation)."""
        hp = ClassifierHyperparameters()
        t = _build_transforms(hp, training=True)
        transform_names = [type(tr).__name__ for tr in t.transforms]
        assert "RandomErasing" in transform_names

    def test_val_transforms_no_random_rotation(self) -> None:
        """Validation transforms do not include RandomRotation."""
        hp = ClassifierHyperparameters()
        t = _build_transforms(hp, training=False)
        transform_names = [type(tr).__name__ for tr in t.transforms]
        assert "RandomRotation" not in transform_names

    def test_val_transforms_no_color_jitter(self) -> None:
        """Validation transforms do not include ColorJitter."""
        hp = ClassifierHyperparameters()
        t = _build_transforms(hp, training=False)
        transform_names = [type(tr).__name__ for tr in t.transforms]
        assert "ColorJitter" not in transform_names

    def test_val_transforms_include_normalize(self) -> None:
        """Validation transforms include Normalize."""
        hp = ClassifierHyperparameters()
        t = _build_transforms(hp, training=False)
        transform_names = [type(tr).__name__ for tr in t.transforms]
        assert "Normalize" in transform_names

    def test_training_transforms_include_normalize(self) -> None:
        """Training transforms include Normalize."""
        hp = ClassifierHyperparameters()
        t = _build_transforms(hp, training=True)
        transform_names = [type(tr).__name__ for tr in t.transforms]
        assert "Normalize" in transform_names

    def test_imagenet_normalization_values(self) -> None:
        """Normalize uses ImageNet mean/std values."""
        hp = ClassifierHyperparameters()
        t = _build_transforms(hp, training=False)
        normalize = next(tr for tr in t.transforms if isinstance(tr, Normalize))
        assert normalize.mean == list(IMAGENET_MEAN)
        assert normalize.std == list(IMAGENET_STD)


# ---------------------------------------------------------------------------
# TestBuildDataLoaders
# ---------------------------------------------------------------------------


class TestBuildDataLoaders:
    """Tests for _build_data_loaders helper."""

    def test_returns_two_data_loaders(
        self, tmp_dataset: ClassificationDatasetConfig
    ) -> None:
        """_build_data_loaders returns (train_loader, val_loader)."""
        hp = ClassifierHyperparameters(batch_size=4, epochs=1)
        train_loader, val_loader = _build_data_loaders(tmp_dataset, hp)
        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)

    def test_train_loader_uses_training_split(
        self, tmp_dataset: ClassificationDatasetConfig
    ) -> None:
        """_build_data_loaders uses dataset['train'] path for train loader."""
        hp = ClassifierHyperparameters(batch_size=4)
        train_loader, _ = _build_data_loaders(tmp_dataset, hp)
        # ImageFolder root should match dataset train path
        assert str(tmp_dataset["train"]) in str(train_loader.dataset.root)  # type: ignore[attr-defined]

    def test_val_loader_uses_val_split(
        self, tmp_dataset: ClassificationDatasetConfig
    ) -> None:
        """_build_data_loaders uses dataset['val'] path for val loader."""
        hp = ClassifierHyperparameters(batch_size=4)
        _, val_loader = _build_data_loaders(tmp_dataset, hp)
        assert str(tmp_dataset["val"]) in str(val_loader.dataset.root)  # type: ignore[attr-defined]

    def test_batch_size_respected(
        self, tmp_dataset: ClassificationDatasetConfig
    ) -> None:
        """_build_data_loaders uses hp.batch_size."""
        hp = ClassifierHyperparameters(batch_size=4)
        train_loader, _ = _build_data_loaders(tmp_dataset, hp)
        assert train_loader.batch_size == 4


# ---------------------------------------------------------------------------
# TestBuildOptimizer
# ---------------------------------------------------------------------------


class TestBuildOptimizer:
    """Tests for _build_optimizer helper."""

    def test_adam_optimizer_created(self, minimal_resnet: nn.Module) -> None:
        """_build_optimizer creates Adam optimizer when hp.optimizer='Adam'."""
        hp = ClassifierHyperparameters(optimizer="Adam", learning_rate=1e-3)
        opt = _build_optimizer(minimal_resnet, hp)
        assert isinstance(opt, torch.optim.Adam)

    def test_adamw_optimizer_created(self, minimal_resnet: nn.Module) -> None:
        """_build_optimizer creates AdamW optimizer when hp.optimizer='AdamW'."""
        hp = ClassifierHyperparameters(optimizer="AdamW", learning_rate=1e-3)
        opt = _build_optimizer(minimal_resnet, hp)
        assert isinstance(opt, torch.optim.AdamW)

    def test_sgd_optimizer_created(self, minimal_resnet: nn.Module) -> None:
        """_build_optimizer creates SGD optimizer when hp.optimizer='SGD'."""
        hp = ClassifierHyperparameters(optimizer="SGD", learning_rate=1e-2)
        opt = _build_optimizer(minimal_resnet, hp)
        assert isinstance(opt, torch.optim.SGD)

    def test_unsupported_optimizer_raises_training_run_error(
        self, minimal_resnet: nn.Module
    ) -> None:
        """_build_optimizer raises TrainingRunError for unsupported optimizer."""
        # Bypass Pydantic validation by patching
        hp = ClassifierHyperparameters(optimizer="Adam")
        object.__setattr__(hp, "optimizer", "RMSprop")
        with pytest.raises(TrainingRunError):
            _build_optimizer(minimal_resnet, hp)


# ---------------------------------------------------------------------------
# TestBuildScheduler
# ---------------------------------------------------------------------------


class TestBuildScheduler:
    """Tests for _build_scheduler helper."""

    def test_step_lr_created(self, minimal_resnet: nn.Module) -> None:
        """_build_scheduler creates StepLR when hp.scheduler='StepLR'."""
        hp = ClassifierHyperparameters(scheduler="StepLR")
        opt = torch.optim.Adam(minimal_resnet.parameters(), lr=1e-3)
        sched = _build_scheduler(opt, hp)
        assert isinstance(sched, StepLR)

    def test_cosine_annealing_lr_created(self, minimal_resnet: nn.Module) -> None:
        """_build_scheduler creates CosineAnnealingLR when hp.scheduler='CosineAnnealingLR'."""
        hp = ClassifierHyperparameters(scheduler="CosineAnnealingLR")
        opt = torch.optim.Adam(minimal_resnet.parameters(), lr=1e-3)
        sched = _build_scheduler(opt, hp)
        assert isinstance(sched, CosineAnnealingLR)

    def test_none_scheduler_returns_none(self, minimal_resnet: nn.Module) -> None:
        """_build_scheduler returns None when hp.scheduler='none'."""
        hp = ClassifierHyperparameters(scheduler="none")
        opt = torch.optim.Adam(minimal_resnet.parameters(), lr=1e-3)
        sched = _build_scheduler(opt, hp)
        assert sched is None


# ---------------------------------------------------------------------------
# TestComputeEce
# ---------------------------------------------------------------------------


class TestComputeEce:
    """Tests for _compute_ece helper."""

    def test_perfect_calibration_near_zero(self) -> None:
        """ECE is near 0 when confidence perfectly matches accuracy."""
        n = 100
        # All predicted prob = 1.0, all correct => ECE ≈ 0
        probs = torch.zeros(n, 6)
        labels = torch.zeros(n, dtype=torch.long)
        probs[:, 0] = 1.0  # Confident on class 0
        ece = _compute_ece(probs, labels, n_bins=15)
        assert ece == pytest.approx(0.0, abs=1e-6)

    def test_random_predictions_positive_ece(self) -> None:
        """ECE is positive for random predictions."""
        torch.manual_seed(42)
        n = 200
        probs = torch.softmax(torch.randn(n, 6), dim=1)
        labels = torch.randint(0, 6, (n,))
        ece = _compute_ece(probs, labels, n_bins=15)
        assert ece >= 0.0

    def test_ece_bounded_between_zero_and_one(self) -> None:
        """ECE is always in [0, 1]."""
        torch.manual_seed(99)
        probs = torch.softmax(torch.randn(50, 6), dim=1)
        labels = torch.randint(0, 6, (50,))
        ece = _compute_ece(probs, labels, n_bins=10)
        assert 0.0 <= ece <= 1.0

    def test_single_sample_does_not_crash(self) -> None:
        """_compute_ece handles single sample without error."""
        probs = torch.softmax(torch.randn(1, 6), dim=1)
        labels = torch.zeros(1, dtype=torch.long)
        ece = _compute_ece(probs, labels, n_bins=15)
        assert isinstance(ece, float)

    def test_all_in_one_bin(self) -> None:
        """_compute_ece handles case where all predictions fall in one bin."""
        n = 30
        # All confident on class 0, all correct
        probs = torch.zeros(n, 6)
        probs[:, 0] = 1.0
        labels = torch.zeros(n, dtype=torch.long)
        ece = _compute_ece(probs, labels, n_bins=15)
        assert ece == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# TestRunTrainEpoch
# ---------------------------------------------------------------------------


class TestRunTrainEpoch:
    """Tests for _run_train_epoch helper."""

    def test_returns_loss_and_accuracy(self, minimal_resnet: nn.Module) -> None:
        """_run_train_epoch returns (avg_loss, top1_accuracy) as floats."""
        device = torch.device("cpu")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(minimal_resnet.parameters(), lr=1e-3)

        # Create tiny fake DataLoader
        fake_images = torch.randn(4, 3, 64, 64)
        fake_labels = torch.zeros(4, dtype=torch.long)
        fake_loader: Any = [(fake_images, fake_labels)]

        loss, acc = _run_train_epoch(
            minimal_resnet, fake_loader, criterion, optimizer, device
        )
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert loss >= 0.0
        assert 0.0 <= acc <= 1.0

    def test_model_in_train_mode_during_epoch(self, minimal_resnet: nn.Module) -> None:
        """_run_train_epoch sets model to train mode."""
        minimal_resnet.eval()  # Start in eval
        device = torch.device("cpu")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(minimal_resnet.parameters(), lr=1e-3)

        fake_images = torch.randn(2, 3, 64, 64)
        fake_labels = torch.zeros(2, dtype=torch.long)
        fake_loader: Any = [(fake_images, fake_labels)]

        _run_train_epoch(minimal_resnet, fake_loader, criterion, optimizer, device)
        # After epoch the model should be in train mode (it was set at start)
        assert minimal_resnet.training

    def test_optimizer_step_called(self, minimal_resnet: nn.Module) -> None:
        """_run_train_epoch calls optimizer.step() at least once."""
        device = torch.device("cpu")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(minimal_resnet.parameters(), lr=1e-3)

        # Wrap step to track calls
        original_step = optimizer.step
        step_calls: list[int] = []

        def _counting_step() -> None:
            step_calls.append(1)
            original_step()

        optimizer.step = _counting_step  # type: ignore[method-assign,assignment]

        fake_images = torch.randn(4, 3, 64, 64)
        fake_labels = torch.zeros(4, dtype=torch.long)
        fake_loader: Any = [(fake_images, fake_labels)]

        _run_train_epoch(minimal_resnet, fake_loader, criterion, optimizer, device)
        assert len(step_calls) >= 1


# ---------------------------------------------------------------------------
# TestRunValEpoch
# ---------------------------------------------------------------------------


class TestRunValEpoch:
    """Tests for _run_val_epoch helper."""

    def test_returns_loss_accuracy_probs_labels(
        self, minimal_resnet: nn.Module
    ) -> None:
        """_run_val_epoch returns (loss, acc, probs, labels) tuple."""
        device = torch.device("cpu")
        criterion = nn.CrossEntropyLoss()

        fake_images = torch.randn(4, 3, 64, 64)
        fake_labels = torch.zeros(4, dtype=torch.long)
        fake_loader: Any = [(fake_images, fake_labels)]

        loss, acc, probs, labels = _run_val_epoch(
            minimal_resnet, fake_loader, criterion, device
        )
        assert isinstance(loss, float)
        assert isinstance(acc, float)
        assert isinstance(probs, torch.Tensor)
        assert isinstance(labels, torch.Tensor)
        assert probs.shape[1] == 6  # 6 classes

    def test_model_in_eval_mode(self, minimal_resnet: nn.Module) -> None:
        """_run_val_epoch sets model to eval mode."""
        minimal_resnet.train()  # Start in train
        device = torch.device("cpu")
        criterion = nn.CrossEntropyLoss()

        fake_images = torch.randn(2, 3, 64, 64)
        fake_labels = torch.zeros(2, dtype=torch.long)
        fake_loader: Any = [(fake_images, fake_labels)]

        _run_val_epoch(minimal_resnet, fake_loader, criterion, device)
        assert not minimal_resnet.training

    def test_no_gradients_computed(self, minimal_resnet: nn.Module) -> None:
        """_run_val_epoch disables gradient computation (no_grad context)."""
        device = torch.device("cpu")
        criterion = nn.CrossEntropyLoss()

        called_no_grad = []

        original_no_grad = torch.no_grad

        def patched_no_grad() -> Any:
            called_no_grad.append(True)
            return original_no_grad()

        fake_images = torch.randn(2, 3, 64, 64)
        fake_labels = torch.zeros(2, dtype=torch.long)
        fake_loader: Any = [(fake_images, fake_labels)]

        with patch("torch.no_grad", patched_no_grad):
            _run_val_epoch(minimal_resnet, fake_loader, criterion, device)
        assert len(called_no_grad) >= 1


# ---------------------------------------------------------------------------
# TestBuildMetadata
# ---------------------------------------------------------------------------


class TestBuildMetadata:
    """Tests for _build_metadata helper."""

    def test_contains_required_keys(self) -> None:
        """_build_metadata returns dict with all required top-level keys."""
        metrics = ClassificationMetrics(
            top1_accuracy=0.85, val_loss=0.33, ece=0.04, best_epoch=15
        )
        hp = ClassifierHyperparameters()
        meta = _build_metadata(
            version="v20260222_120000",
            architecture="resnet18",
            trained_at="2026-02-22T12:00:00Z",
            git_commit="abc1234",
            dataset_root="data/hold_classification",
            hyperparameters=hp,
            metrics=metrics,
        )
        assert "version" in meta
        assert "architecture" in meta
        assert "trained_at" in meta
        assert "git_commit" in meta
        assert "dataset_root" in meta
        assert "hyperparameters" in meta
        assert "metrics" in meta

    def test_metadata_values_match_inputs(self) -> None:
        """_build_metadata stores exact values passed in."""
        metrics = ClassificationMetrics(
            top1_accuracy=0.9, val_loss=0.2, ece=0.02, best_epoch=20
        )
        hp = ClassifierHyperparameters(epochs=25)
        meta = _build_metadata(
            version="v20260222_120000",
            architecture="mobilenet_v3_small",
            trained_at="2026-02-22T12:00:00Z",
            git_commit=None,
            dataset_root="data/holds",
            hyperparameters=hp,
            metrics=metrics,
        )
        assert meta["version"] == "v20260222_120000"
        assert meta["architecture"] == "mobilenet_v3_small"
        assert meta["git_commit"] is None
        assert isinstance(meta["hyperparameters"], dict)
        assert meta["hyperparameters"]["epochs"] == 25

    def test_metadata_is_json_serializable(self) -> None:
        """_build_metadata returns a JSON-serializable dict."""
        metrics = ClassificationMetrics(
            top1_accuracy=0.5, val_loss=1.0, ece=0.1, best_epoch=0
        )
        hp = ClassifierHyperparameters()
        meta = _build_metadata(
            version="v20260222_000000",
            architecture="resnet18",
            trained_at="2026-02-22T00:00:00Z",
            git_commit=None,
            dataset_root="data/holds",
            hyperparameters=hp,
            metrics=metrics,
        )
        json_str = json.dumps(meta)
        assert len(json_str) > 0


# ---------------------------------------------------------------------------
# TestSaveArtifacts
# ---------------------------------------------------------------------------


class TestSaveArtifacts:
    """Tests for _save_artifacts helper."""

    def _create_state_dict(self) -> dict[str, Any]:
        """Create a minimal fake state dict."""
        return {"fc.weight": torch.zeros(6, 512), "fc.bias": torch.zeros(6)}

    def test_creates_versioned_directory_structure(self, tmp_path: Path) -> None:
        """_save_artifacts creates <output_dir>/<version>/weights/ structure."""
        best_state = self._create_state_dict()
        last_state = self._create_state_dict()
        meta = {"version": "v20260222_120000"}
        output_dir = tmp_path / "models" / "classification"
        output_dir.mkdir(parents=True)

        best_path, last_path, meta_path = _save_artifacts(
            best_state=best_state,
            last_state=last_state,
            metadata=meta,
            out_dir=output_dir,
            version="v20260222_120000",
        )

        assert best_path.exists()
        assert last_path.exists()
        assert meta_path.exists()
        assert best_path.name == "best.pt"
        assert last_path.name == "last.pt"
        assert meta_path.name == "metadata.json"

    def test_metadata_json_content_correct(self, tmp_path: Path) -> None:
        """_save_artifacts writes valid JSON to metadata.json."""
        best_state = self._create_state_dict()
        last_state = self._create_state_dict()
        meta = {"version": "v20260222_120000", "architecture": "resnet18"}
        output_dir = tmp_path / "models"
        output_dir.mkdir()

        _, _, meta_path = _save_artifacts(
            best_state=best_state,
            last_state=last_state,
            metadata=meta,
            out_dir=output_dir,
            version="v20260222_120000",
        )

        with open(meta_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        assert loaded["version"] == "v20260222_120000"
        assert loaded["architecture"] == "resnet18"

    def test_weights_dir_inside_versioned_dir(self, tmp_path: Path) -> None:
        """_save_artifacts places weights under <version>/weights/."""
        best_state = self._create_state_dict()
        last_state = self._create_state_dict()
        meta = {"version": "v20260222_120000"}
        output_dir = tmp_path / "models"
        output_dir.mkdir()

        best_path, _, _ = _save_artifacts(
            best_state=best_state,
            last_state=last_state,
            metadata=meta,
            out_dir=output_dir,
            version="v20260222_120000",
        )

        assert best_path.parent.name == "weights"
        assert best_path.parent.parent.name == "v20260222_120000"

    def test_raises_model_artifact_error_on_io_error(self, tmp_path: Path) -> None:
        """_save_artifacts raises ModelArtifactError on I/O failure."""
        best_state = self._create_state_dict()
        last_state = self._create_state_dict()
        meta = {"version": "v20260222_120000"}
        # Use a file as output_dir (not a directory) to trigger IOError
        fake_file = tmp_path / "not_a_dir"
        fake_file.write_text("x")

        with pytest.raises(ModelArtifactError):
            _save_artifacts(
                best_state=best_state,
                last_state=last_state,
                metadata=meta,
                out_dir=fake_file,
                version="v20260222_120000",
            )


# ---------------------------------------------------------------------------
# TestTrainHoldClassifier (integration)
# ---------------------------------------------------------------------------


class TestTrainHoldClassifier:  # pylint: disable=too-many-public-methods
    """Integration tests for train_hold_classifier."""

    def _patch_and_run(
        self,
        tmp_path: Path,
        dataset: ClassificationDatasetConfig,
        hp: ClassifierHyperparameters | None = None,
    ) -> "ClassificationTrainingResult":
        """Helper: run train_hold_classifier with patched heavy dependencies."""
        device = torch.device("cpu")

        # Fake model
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 6)

        # Fake epoch outputs
        fake_probs = torch.softmax(torch.randn(12, 6), dim=1)
        fake_labels = torch.randint(0, 6, (12,))

        with (
            patch(
                "src.training.train_classification.build_hold_classifier"
            ) as mock_build,
            patch(
                "src.training.train_classification._build_data_loaders"
            ) as mock_loaders,
            patch(
                "src.training.train_classification._run_train_epoch"
            ) as mock_train_ep,
            patch("src.training.train_classification._run_val_epoch") as mock_val_ep,
            patch("src.training.train_classification._get_device") as mock_device,
        ):
            mock_build.return_value = {
                "model": model,
                "architecture": "resnet18",
                "num_classes": 6,
                "input_size": 224,
                "pretrained": True,
            }

            # Fake DataLoaders
            fake_loader = MagicMock()
            mock_loaders.return_value = (fake_loader, fake_loader)

            mock_train_ep.return_value = (0.5, 0.75)
            mock_val_ep.return_value = (0.4, 0.80, fake_probs, fake_labels)
            mock_device.return_value = device

            return train_hold_classifier(
                dataset=dataset,
                dataset_root=tmp_path / "data",
                hyperparameters=hp,
                output_dir=tmp_path / "models",
            )

    def test_returns_classification_training_result(self, tmp_path: Path) -> None:
        """train_hold_classifier returns ClassificationTrainingResult."""
        dataset = _make_classification_dataset_config(
            train=tmp_path / "train", val=tmp_path / "val"
        )
        result = self._patch_and_run(tmp_path, dataset)
        assert isinstance(result, ClassificationTrainingResult)

    def test_artifact_paths_exist(self, tmp_path: Path) -> None:
        """train_hold_classifier saves best.pt, last.pt, and metadata.json."""
        dataset = _make_classification_dataset_config(
            train=tmp_path / "train", val=tmp_path / "val"
        )
        result = self._patch_and_run(tmp_path, dataset)
        assert result.best_weights_path.exists()
        assert result.last_weights_path.exists()
        assert result.metadata_path.exists()

    def test_version_matches_format(self, tmp_path: Path) -> None:
        """train_hold_classifier version string matches v\\d{8}_\\d{6}."""
        dataset = _make_classification_dataset_config(
            train=tmp_path / "train", val=tmp_path / "val"
        )
        result = self._patch_and_run(tmp_path, dataset)
        assert re.match(r"^v\d{8}_\d{6}$", result.version)

    def test_uses_default_hyperparameters_when_none(self, tmp_path: Path) -> None:
        """train_hold_classifier uses default hyperparams when None passed."""
        dataset = _make_classification_dataset_config(
            train=tmp_path / "train", val=tmp_path / "val"
        )
        result = self._patch_and_run(tmp_path, dataset, hp=None)
        assert "epochs" in result.hyperparameters

    def test_result_architecture_matches_hyperparameters(self, tmp_path: Path) -> None:
        """train_hold_classifier result.architecture matches hp.architecture."""
        dataset = _make_classification_dataset_config(
            train=tmp_path / "train", val=tmp_path / "val"
        )
        result = self._patch_and_run(tmp_path, dataset)
        assert result.architecture == "resnet18"

    def test_metrics_populated(self, tmp_path: Path) -> None:
        """train_hold_classifier populates metrics from epoch results."""
        dataset = _make_classification_dataset_config(
            train=tmp_path / "train", val=tmp_path / "val"
        )
        result = self._patch_and_run(tmp_path, dataset)
        assert isinstance(result.metrics, ClassificationMetrics)
        assert result.metrics.top1_accuracy >= 0.0
        assert result.metrics.val_loss >= 0.0

    def test_dataset_root_stored_in_result(self, tmp_path: Path) -> None:
        """train_hold_classifier stores dataset_root in result."""
        dataset = _make_classification_dataset_config(
            train=tmp_path / "train", val=tmp_path / "val"
        )
        result = self._patch_and_run(tmp_path, dataset)
        assert result.dataset_root is not None
