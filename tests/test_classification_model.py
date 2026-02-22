# pylint: disable=redefined-outer-name  # standard pytest fixture pattern
"""Tests for the hold classifier model definition module.

Tests cover:
  - Module-level constants (DEFAULT_ARCHITECTURE, INPUT_SIZE, VALID_ARCHITECTURES)
  - ClassifierHyperparameters Pydantic model validation
  - build_hold_classifier() constructor and architecture dispatch
  - get_default_hyperparameters() factory
  - load_hyperparameters_from_file() YAML loader
"""

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from torch import nn
import torchvision.models as tv_models
from pydantic import ValidationError

from src.training import HOLD_CLASS_COUNT, HOLD_CLASSES
from src.training.classification_model import (
    DEFAULT_ARCHITECTURE,
    INPUT_SIZE,
    VALID_ARCHITECTURES,
    ClassifierConfig,
    ClassifierHyperparameters,
    build_hold_classifier,
    get_default_hyperparameters,
    load_hyperparameters_from_file,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_hyperparams_yaml(tmp_path: Path) -> Path:
    """Create a YAML file with valid non-default hyperparameters.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.

    Returns:
        Path to the created YAML configuration file.
    """
    yaml_content = textwrap.dedent(
        """\
        architecture: mobilenet_v3_small
        epochs: 50
        batch_size: 16
        learning_rate: 0.0005
        optimizer: AdamW
        """
    )
    yaml_file = tmp_path / "valid_config.yaml"
    yaml_file.write_text(yaml_content)
    return yaml_file


@pytest.fixture
def invalid_hyperparams_yaml(tmp_path: Path) -> Path:
    """Create a YAML file with invalid hyperparameter values.

    Args:
        tmp_path: Pytest fixture providing a temporary directory path.

    Returns:
        Path to the created YAML configuration file.
    """
    yaml_content = textwrap.dedent(
        """\
        architecture: vgg16
        epochs: -1
        optimizer: RMSProp
        """
    )
    yaml_file = tmp_path / "invalid_config.yaml"
    yaml_file.write_text(yaml_content)
    return yaml_file


# ---------------------------------------------------------------------------
# TestModuleConstants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    """Tests for module-level constants."""

    def test_default_architecture_is_resnet18(self) -> None:
        """DEFAULT_ARCHITECTURE should be 'resnet18'."""
        assert DEFAULT_ARCHITECTURE == "resnet18"

    def test_default_architecture_is_in_valid_architectures(self) -> None:
        """DEFAULT_ARCHITECTURE must be a member of VALID_ARCHITECTURES."""
        assert DEFAULT_ARCHITECTURE in VALID_ARCHITECTURES

    def test_input_size_is_224(self) -> None:
        """INPUT_SIZE must be 224 (standard ImageNet input)."""
        assert INPUT_SIZE == 224

    def test_valid_architectures_exact_set(self) -> None:
        """VALID_ARCHITECTURES must contain exactly the three supported backbones."""
        assert VALID_ARCHITECTURES == (
            "resnet18",
            "mobilenet_v3_small",
            "mobilenet_v3_large",
        )

    def test_valid_architectures_is_tuple(self) -> None:
        """VALID_ARCHITECTURES must be a tuple (immutable constant per project policy)."""
        assert isinstance(VALID_ARCHITECTURES, tuple)


# ---------------------------------------------------------------------------
# TestClassifierHyperparameters
# ---------------------------------------------------------------------------


class TestClassifierHyperparameters:  # pylint: disable=too-many-public-methods
    """Tests for ClassifierHyperparameters Pydantic model."""

    # --- Default values ---

    def test_default_architecture(self) -> None:
        """Default architecture is resnet18."""
        hp = ClassifierHyperparameters()
        assert hp.architecture == "resnet18"

    def test_default_epochs(self) -> None:
        """Default epochs is 30."""
        hp = ClassifierHyperparameters()
        assert hp.epochs == 30

    def test_default_batch_size(self) -> None:
        """Default batch_size is 32."""
        hp = ClassifierHyperparameters()
        assert hp.batch_size == 32

    def test_default_learning_rate(self) -> None:
        """Default learning_rate is 1e-3."""
        hp = ClassifierHyperparameters()
        assert hp.learning_rate == pytest.approx(1e-3)

    def test_default_optimizer(self) -> None:
        """Default optimizer is Adam."""
        hp = ClassifierHyperparameters()
        assert hp.optimizer == "Adam"

    def test_default_pretrained(self) -> None:
        """Default pretrained is True."""
        hp = ClassifierHyperparameters()
        assert hp.pretrained is True

    def test_default_label_smoothing(self) -> None:
        """Default label_smoothing is 0.1."""
        hp = ClassifierHyperparameters()
        assert hp.label_smoothing == pytest.approx(0.1)

    def test_default_num_classes(self) -> None:
        """Default num_classes equals HOLD_CLASS_COUNT (6)."""
        hp = ClassifierHyperparameters()
        assert hp.num_classes == HOLD_CLASS_COUNT

    def test_default_input_size(self) -> None:
        """Default input_size equals INPUT_SIZE (224)."""
        hp = ClassifierHyperparameters()
        assert hp.input_size == INPUT_SIZE

    def test_default_scheduler(self) -> None:
        """Default scheduler is StepLR."""
        hp = ClassifierHyperparameters()
        assert hp.scheduler == "StepLR"

    def test_default_weight_decay(self) -> None:
        """Default weight_decay is 1e-4."""
        hp = ClassifierHyperparameters()
        assert hp.weight_decay == pytest.approx(1e-4)

    def test_default_dropout_rate(self) -> None:
        """Default dropout_rate is 0.2."""
        hp = ClassifierHyperparameters()
        assert hp.dropout_rate == pytest.approx(0.2)

    # --- Valid architecture variants ---

    @pytest.mark.parametrize(
        "arch", ["resnet18", "mobilenet_v3_small", "mobilenet_v3_large"]
    )
    def test_valid_architectures_accepted(self, arch: str) -> None:
        """All supported architectures are accepted without error."""
        hp = ClassifierHyperparameters(architecture=arch)
        assert hp.architecture == arch

    # --- Invalid architecture ---

    def test_invalid_architecture_raises_validation_error(self) -> None:
        """Unsupported architecture (e.g. vgg16) raises ValidationError."""
        with pytest.raises(ValidationError):
            ClassifierHyperparameters(architecture="vgg16")

    # --- Valid optimizers ---

    @pytest.mark.parametrize("opt", ["SGD", "Adam", "AdamW"])
    def test_valid_optimizers_accepted(self, opt: str) -> None:
        """All supported optimizers are accepted without error."""
        hp = ClassifierHyperparameters(optimizer=opt)
        assert hp.optimizer == opt

    # --- Invalid optimizer ---

    def test_invalid_optimizer_raises_validation_error(self) -> None:
        """Unsupported optimizer (e.g. RMSProp) raises ValidationError."""
        with pytest.raises(ValidationError):
            ClassifierHyperparameters(optimizer="RMSProp")

    # --- Valid schedulers ---

    @pytest.mark.parametrize("sched", ["StepLR", "CosineAnnealingLR", "none"])
    def test_valid_schedulers_accepted(self, sched: str) -> None:
        """All supported schedulers are accepted without error."""
        hp = ClassifierHyperparameters(scheduler=sched)
        assert hp.scheduler == sched

    # --- Invalid scheduler ---

    def test_invalid_scheduler_raises_validation_error(self) -> None:
        """Unsupported scheduler (e.g. ReduceLROnPlateau) raises ValidationError."""
        with pytest.raises(ValidationError):
            ClassifierHyperparameters(scheduler="ReduceLROnPlateau")

    # --- Numeric bounds ---

    def test_epochs_zero_raises_validation_error(self) -> None:
        """epochs=0 is below ge=1 and must raise ValidationError."""
        with pytest.raises(ValidationError):
            ClassifierHyperparameters(epochs=0)

    def test_learning_rate_zero_raises_validation_error(self) -> None:
        """learning_rate=0.0 violates gt=0.0 and must raise ValidationError."""
        with pytest.raises(ValidationError):
            ClassifierHyperparameters(learning_rate=0.0)

    def test_batch_size_zero_raises_validation_error(self) -> None:
        """batch_size=0 is below ge=1 and must raise ValidationError."""
        with pytest.raises(ValidationError):
            ClassifierHyperparameters(batch_size=0)

    def test_label_smoothing_one_raises_validation_error(self) -> None:
        """label_smoothing=1.0 violates lt=1.0 and must raise ValidationError."""
        with pytest.raises(ValidationError):
            ClassifierHyperparameters(label_smoothing=1.0)

    def test_dropout_rate_one_raises_validation_error(self) -> None:
        """dropout_rate=1.0 violates lt=1.0 and must raise ValidationError."""
        with pytest.raises(ValidationError):
            ClassifierHyperparameters(dropout_rate=1.0)

    # --- to_dict ---

    def test_to_dict_returns_dict(self) -> None:
        """to_dict() must return a plain dict."""
        hp = ClassifierHyperparameters()
        assert isinstance(hp.to_dict(), dict)

    def test_to_dict_contains_architecture(self) -> None:
        """to_dict() output must contain 'architecture' key."""
        hp = ClassifierHyperparameters()
        assert "architecture" in hp.to_dict()

    def test_to_dict_values_match_fields(self) -> None:
        """to_dict() values must match field values."""
        hp = ClassifierHyperparameters(epochs=42, batch_size=8)
        d = hp.to_dict()
        assert d["epochs"] == 42
        assert d["batch_size"] == 8

    def test_to_dict_uses_field_name_not_alias(self) -> None:
        """to_dict() must use 'learning_rate', not an alias like 'lr0'."""
        hp = ClassifierHyperparameters()
        d = hp.to_dict()
        assert "learning_rate" in d
        assert "lr0" not in d


# ---------------------------------------------------------------------------
# TestBuildHoldClassifier
# ---------------------------------------------------------------------------


class TestBuildHoldClassifier:
    """Tests for build_hold_classifier() constructor function."""

    @patch("src.training.classification_model.models.resnet18")
    def test_build_default_classifier_calls_resnet18(
        self, mock_resnet18: MagicMock
    ) -> None:
        """build_hold_classifier() with defaults must call models.resnet18."""
        mock_model = MagicMock()
        mock_model.fc = MagicMock()
        mock_resnet18.return_value = mock_model
        config = build_hold_classifier()
        mock_resnet18.assert_called_once()
        assert config["model"] is mock_model

    @patch("src.training.classification_model.models.resnet18")
    def test_build_resnet18_architecture(self, mock_resnet18: MagicMock) -> None:
        """Explicit resnet18 architecture must call models.resnet18."""
        mock_model = MagicMock()
        mock_model.fc = MagicMock()
        mock_resnet18.return_value = mock_model
        hp = ClassifierHyperparameters(architecture="resnet18")
        config = build_hold_classifier(hp)
        mock_resnet18.assert_called_once()
        assert config["architecture"] == "resnet18"

    @patch("src.training.classification_model.models.mobilenet_v3_small")
    def test_build_mobilenet_v3_small_architecture(
        self, mock_mobilenet_small: MagicMock
    ) -> None:
        """mobilenet_v3_small architecture must call models.mobilenet_v3_small."""
        mock_model = MagicMock()
        mock_mobilenet_small.return_value = mock_model
        hp = ClassifierHyperparameters(architecture="mobilenet_v3_small")
        config = build_hold_classifier(hp)
        mock_mobilenet_small.assert_called_once()
        assert config["architecture"] == "mobilenet_v3_small"

    @patch("src.training.classification_model.models.mobilenet_v3_large")
    def test_build_mobilenet_v3_large_architecture(
        self, mock_mobilenet_large: MagicMock
    ) -> None:
        """mobilenet_v3_large architecture must call models.mobilenet_v3_large."""
        mock_model = MagicMock()
        mock_mobilenet_large.return_value = mock_model
        hp = ClassifierHyperparameters(architecture="mobilenet_v3_large")
        config = build_hold_classifier(hp)
        mock_mobilenet_large.assert_called_once()
        assert config["architecture"] == "mobilenet_v3_large"

    @patch("src.training.classification_model.models.resnet18")
    def test_pretrained_true_passes_default_weights(
        self, mock_resnet18: MagicMock
    ) -> None:
        """pretrained=True must pass ResNet18_Weights.DEFAULT to models.resnet18."""
        mock_model = MagicMock()
        mock_model.fc = MagicMock()
        mock_resnet18.return_value = mock_model
        hp = ClassifierHyperparameters(architecture="resnet18", pretrained=True)
        build_hold_classifier(hp)
        call_kwargs = mock_resnet18.call_args
        assert call_kwargs is not None
        _, kwargs = call_kwargs
        assert kwargs.get("weights") == tv_models.ResNet18_Weights.DEFAULT

    @patch("src.training.classification_model.models.resnet18")
    def test_pretrained_false_passes_none_weights(
        self, mock_resnet18: MagicMock
    ) -> None:
        """pretrained=False must pass weights=None to models.resnet18."""
        mock_model = MagicMock()
        mock_model.fc = MagicMock()
        mock_resnet18.return_value = mock_model
        hp = ClassifierHyperparameters(architecture="resnet18", pretrained=False)
        build_hold_classifier(hp)
        call_kwargs = mock_resnet18.call_args
        assert call_kwargs is not None
        _, kwargs = call_kwargs
        assert kwargs.get("weights") is None

    @patch("src.training.classification_model.models.resnet18")
    def test_returned_dict_has_exact_keys(self, mock_resnet18: MagicMock) -> None:
        """ClassifierConfig dict must have exactly the five expected keys."""
        mock_model = MagicMock()
        mock_model.fc = MagicMock()
        mock_resnet18.return_value = mock_model
        config = build_hold_classifier()
        assert set(config.keys()) == {
            "model",
            "architecture",
            "num_classes",
            "input_size",
            "pretrained",
        }

    @patch("src.training.classification_model.models.resnet18")
    def test_num_classes_equals_hold_class_count(
        self, mock_resnet18: MagicMock
    ) -> None:
        """config['num_classes'] must equal HOLD_CLASS_COUNT (6)."""
        mock_model = MagicMock()
        mock_model.fc = MagicMock()
        mock_resnet18.return_value = mock_model
        config = build_hold_classifier()
        assert config["num_classes"] == HOLD_CLASS_COUNT == 6

    @patch("src.training.classification_model.models.resnet18")
    def test_input_size_equals_input_size_constant(
        self, mock_resnet18: MagicMock
    ) -> None:
        """config['input_size'] must equal INPUT_SIZE (224)."""
        mock_model = MagicMock()
        mock_model.fc = MagicMock()
        mock_resnet18.return_value = mock_model
        config = build_hold_classifier()
        assert config["input_size"] == INPUT_SIZE == 224

    def test_wrong_num_classes_raises_value_error(self) -> None:
        """num_classes != HOLD_CLASS_COUNT must raise ValueError."""
        hp = ClassifierHyperparameters(num_classes=3)
        with pytest.raises(ValueError, match="num_classes"):
            build_hold_classifier(hp)

    @patch("src.training.classification_model.models.resnet18")
    def test_none_hyperparameters_uses_defaults(self, mock_resnet18: MagicMock) -> None:
        """Passing hyperparameters=None must use defaults and call resnet18."""
        mock_model = MagicMock()
        mock_model.fc = MagicMock()
        mock_resnet18.return_value = mock_model
        config = build_hold_classifier(hyperparameters=None)
        mock_resnet18.assert_called_once()
        assert config["architecture"] == "resnet18"

    @patch("src.training.classification_model.models.resnet18")
    def test_pretrained_flag_reflected_in_config(
        self, mock_resnet18: MagicMock
    ) -> None:
        """config['pretrained'] must match the hyperparameter value."""
        mock_model = MagicMock()
        mock_model.fc = MagicMock()
        mock_resnet18.return_value = mock_model
        hp = ClassifierHyperparameters(pretrained=False)
        config = build_hold_classifier(hp)
        assert config["pretrained"] is False

    @patch("src.training.classification_model.models.resnet18")
    def test_config_model_is_returned_backbone(self, mock_resnet18: MagicMock) -> None:
        """config['model'] must be the exact backbone instance returned by torchvision."""
        mock_model = MagicMock()
        mock_model.fc = MagicMock()
        mock_resnet18.return_value = mock_model
        config = build_hold_classifier()
        assert config["model"] is mock_model


# ---------------------------------------------------------------------------
# TestGetDefaultHyperparameters
# ---------------------------------------------------------------------------


class TestGetDefaultHyperparameters:
    """Tests for get_default_hyperparameters() factory."""

    def test_returns_classifier_hyperparameters_instance(self) -> None:
        """get_default_hyperparameters() must return a ClassifierHyperparameters."""
        hp = get_default_hyperparameters()
        assert isinstance(hp, ClassifierHyperparameters)

    def test_default_values_match_constants(self) -> None:
        """Returned instance must have default architecture and input size."""
        hp = get_default_hyperparameters()
        assert hp.architecture == DEFAULT_ARCHITECTURE
        assert hp.input_size == INPUT_SIZE

    def test_num_classes_equals_hold_class_count(self) -> None:
        """Returned instance num_classes must equal HOLD_CLASS_COUNT."""
        hp = get_default_hyperparameters()
        assert hp.num_classes == HOLD_CLASS_COUNT


# ---------------------------------------------------------------------------
# TestLoadHyperparametersFromFile
# ---------------------------------------------------------------------------


class TestLoadHyperparametersFromFile:
    """Tests for load_hyperparameters_from_file() YAML loader."""

    def test_valid_yaml_loaded_correctly(self, valid_hyperparams_yaml: Path) -> None:
        """Valid YAML file must be loaded into a ClassifierHyperparameters instance."""
        hp = load_hyperparameters_from_file(valid_hyperparams_yaml)
        assert isinstance(hp, ClassifierHyperparameters)
        assert hp.architecture == "mobilenet_v3_small"
        assert hp.epochs == 50
        assert hp.batch_size == 16
        assert hp.learning_rate == pytest.approx(5e-4)
        assert hp.optimizer == "AdamW"

    def test_nonexistent_file_raises_file_not_found_error(self, tmp_path: Path) -> None:
        """Non-existent file must raise FileNotFoundError with 'Config file not found'."""
        missing = tmp_path / "does_not_exist.yaml"
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            load_hyperparameters_from_file(missing)

    def test_empty_yaml_returns_defaults(self, tmp_path: Path) -> None:
        """Empty YAML file must return an instance with all default values."""
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("")
        hp = load_hyperparameters_from_file(empty_file)
        assert isinstance(hp, ClassifierHyperparameters)
        assert hp.architecture == DEFAULT_ARCHITECTURE
        assert hp.epochs == 30

    def test_partial_yaml_keeps_defaults_for_omitted_fields(
        self, tmp_path: Path
    ) -> None:
        """YAML with only 'epochs: 75' must keep all other fields at defaults."""
        partial_file = tmp_path / "partial.yaml"
        partial_file.write_text("epochs: 75\n")
        hp = load_hyperparameters_from_file(partial_file)
        assert hp.epochs == 75
        assert hp.architecture == DEFAULT_ARCHITECTURE
        assert hp.batch_size == 32

    def test_invalid_yaml_values_raise_validation_error(
        self, invalid_hyperparams_yaml: Path
    ) -> None:
        """YAML with invalid values (bad architecture, negative epochs) must raise ValidationError."""
        with pytest.raises(ValidationError):
            load_hyperparameters_from_file(invalid_hyperparams_yaml)

    def test_string_path_accepted(self, valid_hyperparams_yaml: Path) -> None:
        """A string path must be accepted (not just Path objects)."""
        hp = load_hyperparameters_from_file(str(valid_hyperparams_yaml))
        assert isinstance(hp, ClassifierHyperparameters)
        assert hp.architecture == "mobilenet_v3_small"

    def test_malformed_yaml_raises_value_error(self, tmp_path: Path) -> None:
        """Syntactically broken YAML must raise ValueError matching 'Failed to parse YAML'."""
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("key: [unclosed bracket\n")
        with pytest.raises(ValueError, match="Failed to parse YAML"):
            load_hyperparameters_from_file(bad_file)


# ---------------------------------------------------------------------------
# ClassifierConfig TypedDict structure sanity check
# ---------------------------------------------------------------------------


class TestClassifierConfigStructure:
    """Sanity check that ClassifierConfig TypedDict keys are importable."""

    def test_classifier_config_is_importable(self) -> None:
        """ClassifierConfig must be importable from the module."""
        assert ClassifierConfig is not None

    def test_hold_classes_accessible_via_training_init(self) -> None:
        """HOLD_CLASSES and HOLD_CLASS_COUNT must be importable via src.training."""
        assert len(HOLD_CLASSES) == HOLD_CLASS_COUNT == 6


# ---------------------------------------------------------------------------
# Integration tests — real backbone verification (pretrained=False, no download)
# ---------------------------------------------------------------------------


class TestRealBackboneIntegration:
    """Integration tests using real torchvision models (pretrained=False).

    These tests verify that the hardcoded ``_*_FEATURES`` constants in
    ``classification_model.py`` match the actual ``in_features`` of the
    final layer for each backbone.  A torchvision version bump that changes
    the architecture layout would be caught here.

    ``pretrained=False`` is used to avoid downloading ImageNet weights in CI.
    """

    def test_resnet18_final_layer_replaced_correctly(self) -> None:
        """Real ResNet-18 backbone must have fc replaced with out_features=6."""
        hp = ClassifierHyperparameters(architecture="resnet18", pretrained=False)
        config = build_hold_classifier(hp)
        fc = config["model"].fc  # type: ignore[union-attr]
        assert isinstance(fc, nn.Linear)
        assert fc.out_features == 6

    def test_mobilenet_v3_small_final_layer_replaced_correctly(self) -> None:
        """Real MobileNetV3-Small backbone must have classifier[-1] replaced with out_features=6."""
        hp = ClassifierHyperparameters(
            architecture="mobilenet_v3_small", pretrained=False
        )
        config = build_hold_classifier(hp)
        last_layer = config["model"].classifier[-1]  # type: ignore[index]
        assert isinstance(last_layer, nn.Linear)
        assert last_layer.out_features == 6

    def test_mobilenet_v3_large_final_layer_replaced_correctly(self) -> None:
        """Real MobileNetV3-Large backbone must have classifier[-1] replaced with out_features=6."""
        hp = ClassifierHyperparameters(
            architecture="mobilenet_v3_large", pretrained=False
        )
        config = build_hold_classifier(hp)
        last_layer = config["model"].classifier[-1]  # type: ignore[index]
        assert isinstance(last_layer, nn.Linear)
        assert last_layer.out_features == 6
