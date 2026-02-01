# PR-3.1: Detection Dataset Schema — Detailed Specification

**Status**: DRAFT (Pending Review)
**Milestone**: 3 — Hold Detection (Pre-Training Phase)
**Dependencies**: None (can run parallel with M1/M2)
**Estimated Effort**: Small

---

## 1. Objective

Create the dataset loading infrastructure for hold/volume detection training that:

- [x] Loads and validates YOLOv8 format detection datasets
- [x] Supports Roboflow exports as primary data source
- [x] Implements dataset versioning metadata
- [x] Enforces simplified class taxonomy: `[hold, volume]`
- [x] Provides clear error messages for malformed datasets

---

## 2. Scope

### In Scope

1. **Module Creation**: Create `src/training/datasets.py`
2. **Core Function**: Implement `load_hold_detection_dataset()`
3. **Validation**: Directory structure, YAML config, label format
4. **Class Taxonomy**: Enforce exactly 2 classes: `hold` and `volume`
5. **Metadata**: Dataset versioning and statistics
6. **Testing**: Comprehensive test coverage (>=85%)

### Out of Scope (Future PRs)

- Detection model definition (PR-3.2)
- Training loop implementation (PR-3.3)
- Detection inference (PR-3.4)
- Classification dataset loader (PR-4.2)
- Actual model training

---

## 3. Dataset Format

### 3.1 YOLOv8 Detection Format

```text
dataset_root/
├── data.yaml              # Dataset configuration
├── train/
│   ├── images/            # Training images (.jpg, .png)
│   └── labels/            # YOLO format labels (.txt)
├── val/
│   ├── images/            # Validation images
│   └── labels/            # YOLO format labels
└── test/                  # Optional test set
    ├── images/
    └── labels/
```

### 3.2 data.yaml Structure

```yaml
# Required fields
train: train           # Path to training directory
val: val               # Path to validation directory
nc: 2                  # Number of classes (must be 2)
names:
  0: hold              # Any climbing hold
  1: volume            # Large structural elements

# Optional fields
test: test             # Path to test directory
dataset_version: rf-climbing-holds-v1
export_format: yolov8
export_date: 2026-01-28
```

### 3.3 YOLO Label Format

Each `.txt` file contains one bounding box per line:

```text
<class_id> <x_center> <y_center> <width> <height>
```

- All values normalized to [0, 1] relative to image dimensions
- `class_id`: 0 (hold) or 1 (volume)
- `x_center`, `y_center`: Center of bounding box
- `width`, `height`: Dimensions of bounding box

**Example** (image_001.txt):
```text
0 0.5 0.3 0.1 0.15
1 0.7 0.6 0.2 0.25
0 0.2 0.8 0.12 0.18
```

---

## 4. Class Taxonomy

### 4.1 Detection Classes (2 classes)

| Class ID | Name | Description |
|----------|------|-------------|
| 0 | `hold` | Any hand/foot climbing hold |
| 1 | `volume` | Large structural elements (cubes, walls) |

### 4.2 Rationale

Per `docs/MODEL_PRETRAIN.md`:

- Geometry matters more than semantic subtype
- Reduces label noise significantly
- Improves detection recall
- Hold type classification happens separately in PR-4.x

### 4.3 Migration from Legacy

The legacy system used 5+ hold types for detection. This is explicitly simplified:

| Legacy | New |
|--------|-----|
| crimp, jug, sloper, pinch, pocket | hold (class 0) |
| volume | volume (class 1) |

---

## 5. File Structure

### New Files

```text
src/
└── training/
    ├── __init__.py           # Package exports (~10 lines)
    ├── datasets.py           # Dataset loading logic (~250 lines)
    └── exceptions.py         # Custom exceptions (~30 lines)

tests/
└── test_training_datasets.py # Dataset tests (~400 lines)
```

### Modified Files

None — this is a new module.

---

## 6. Function Contracts

### 6.1 Main Function

```python
def load_hold_detection_dataset(
    dataset_root: Path | str,
    strict: bool = True
) -> dict[str, Any]:
    """Load and validate a hold detection dataset in YOLOv8 format.

    This function loads datasets exported from Roboflow or other sources
    in YOLOv8 object detection format. It validates the directory structure,
    configuration file, and class taxonomy.

    Args:
        dataset_root: Path to the dataset directory containing data.yaml.
        strict: If True, raise errors for validation failures.
            If False, log warnings and continue where possible.

    Returns:
        Dictionary containing:
            - train: Absolute path to training directory
            - val: Absolute path to validation directory
            - test: Absolute path to test directory (or None)
            - nc: Number of classes (always 2)
            - names: List of class names ["hold", "volume"]
            - train_image_count: Number of training images
            - val_image_count: Number of validation images
            - test_image_count: Number of test images (or 0)
            - version: Dataset version string (if available)
            - metadata: Additional metadata from data.yaml

    Raises:
        DatasetNotFoundError: If dataset_root or data.yaml doesn't exist.
        DatasetValidationError: If dataset structure or config is invalid.
        ClassTaxonomyError: If classes don't match expected [hold, volume].

    Example:
        >>> config = load_hold_detection_dataset("data/climbing_holds")
        >>> print(config["train_image_count"])
        500
        >>> print(config["names"])
        ['hold', 'volume']
    """
```

### 6.2 Validation Functions

```python
def validate_data_yaml(yaml_path: Path) -> dict[str, Any]:
    """Validate and parse data.yaml configuration file.

    Args:
        yaml_path: Path to data.yaml file.

    Returns:
        Parsed YAML content as dictionary.

    Raises:
        DatasetNotFoundError: If file doesn't exist.
        DatasetValidationError: If YAML is invalid or missing required keys.
        ClassTaxonomyError: If class configuration is incorrect.
    """


def validate_directory_structure(
    dataset_root: Path,
    config: dict[str, Any]
) -> None:
    """Validate that required directories exist.

    Args:
        dataset_root: Root directory of the dataset.
        config: Parsed data.yaml configuration.

    Raises:
        DatasetValidationError: If required directories are missing.
    """


def count_dataset_images(split_path: Path) -> int:
    """Count the number of images in a dataset split.

    Args:
        split_path: Path to split directory (e.g., train/).

    Returns:
        Number of image files (.jpg, .jpeg, .png) in images/ subdirectory.
    """
```

### 6.3 Custom Exceptions

```python
class TrainingError(Exception):
    """Base exception for training module errors."""


class DatasetNotFoundError(TrainingError):
    """Raised when dataset or required files are not found."""


class DatasetValidationError(TrainingError):
    """Raised when dataset structure or format is invalid."""


class ClassTaxonomyError(TrainingError):
    """Raised when class configuration doesn't match expected taxonomy."""
```

---

## 7. Validation Rules

### 7.1 Directory Structure

| Check | Requirement | Error Type |
|-------|-------------|------------|
| data.yaml exists | Must exist in dataset_root | DatasetNotFoundError |
| train/ directory | Must exist with images/ subdirectory | DatasetValidationError |
| val/ directory | Must exist with images/ subdirectory | DatasetValidationError |
| test/ directory | Optional | None (skip if missing) |

### 7.2 YAML Configuration

| Check | Requirement | Error Type |
|-------|-------------|------------|
| Valid YAML | Must parse without errors | DatasetValidationError |
| train key | Must be present | DatasetValidationError |
| val key | Must be present | DatasetValidationError |
| nc key | Must be present and equal to 2 | ClassTaxonomyError |
| names key | Must be present as list or dict | ClassTaxonomyError |
| Class names | Must be exactly ["hold", "volume"] | ClassTaxonomyError |

### 7.3 Class Taxonomy

```python
EXPECTED_CLASSES = ["hold", "volume"]
EXPECTED_CLASS_COUNT = 2
```

- `nc` must equal 2
- `names` must contain exactly "hold" and "volume"
- Order: class 0 = hold, class 1 = volume
- Case-sensitive matching

---

## 8. Return Value Schema

```python
{
    # Required paths
    "train": Path("/absolute/path/to/train"),
    "val": Path("/absolute/path/to/val"),
    "test": Path("/absolute/path/to/test") | None,

    # Class configuration
    "nc": 2,
    "names": ["hold", "volume"],

    # Statistics
    "train_image_count": 500,
    "val_image_count": 100,
    "test_image_count": 50,  # 0 if no test set

    # Metadata (optional fields from data.yaml)
    "version": "rf-climbing-holds-v1" | None,
    "metadata": {
        "export_format": "yolov8",
        "export_date": "2026-01-28",
        ...
    }
}
```

---

## 9. Error Handling

### 9.1 Error Categories

| Exception | When Raised | User Message Example |
|-----------|-------------|---------------------|
| DatasetNotFoundError | Path doesn't exist | "Dataset not found at: /path/to/dataset" |
| DatasetNotFoundError | data.yaml missing | "data.yaml not found in: /path/to/dataset" |
| DatasetValidationError | Invalid YAML syntax | "Failed to parse data.yaml: {error}" |
| DatasetValidationError | Missing required key | "Missing required key in data.yaml: 'train'" |
| DatasetValidationError | Missing directory | "Training directory not found: /path/train" |
| ClassTaxonomyError | Wrong class count | "Expected 2 classes, found 5" |
| ClassTaxonomyError | Wrong class names | "Expected classes ['hold', 'volume'], got ['crimp', 'jug', ...]" |

### 9.2 Logging

All validation steps should log:
- INFO: Dataset found, starting validation
- DEBUG: Each validation step passed
- WARNING: Non-critical issues (e.g., no test set)
- ERROR: Validation failures (before raising exception)

---

## 10. Testing Plan

### 10.1 Test Classes

| Test Class | Purpose | Est. Tests |
|------------|---------|------------|
| `TestLoadHoldDetectionDataset` | Main function happy path & errors | 12 |
| `TestValidateDataYaml` | YAML parsing and validation | 10 |
| `TestValidateDirectoryStructure` | Directory checks | 6 |
| `TestCountDatasetImages` | Image counting | 4 |
| `TestClassTaxonomy` | Class name validation | 6 |
| `TestExceptions` | Custom exception classes | 4 |

**Total Estimated Tests**: ~42

### 10.2 Test Scenarios

**Happy Path Tests**:
1. Load valid dataset with train/val only
2. Load valid dataset with train/val/test
3. Load dataset with minimal data.yaml
4. Load dataset with full metadata
5. Count images correctly across splits
6. Return correct paths as absolute Path objects
7. Parse version metadata correctly

**YAML Validation Tests**:
1. Reject missing data.yaml
2. Reject invalid YAML syntax
3. Reject missing 'train' key
4. Reject missing 'val' key
5. Reject missing 'nc' key
6. Reject missing 'names' key
7. Accept names as dict format
8. Accept names as list format
9. Reject extra required keys gracefully
10. Handle empty data.yaml

**Directory Structure Tests**:
1. Reject missing train directory
2. Reject missing val directory
3. Accept missing test directory (optional)
4. Reject missing images subdirectory
5. Handle empty images directory
6. Validate relative paths resolved correctly

**Class Taxonomy Tests**:
1. Reject nc != 2
2. Reject wrong class names
3. Reject missing class 0
4. Reject missing class 1
5. Reject extra classes
6. Accept classes in different order in names dict

**Exception Tests**:
1. DatasetNotFoundError has correct message
2. DatasetValidationError has correct message
3. ClassTaxonomyError has correct message
4. All exceptions inherit from TrainingError

### 10.3 Test Fixtures

```python
@pytest.fixture
def valid_dataset(tmp_path: Path) -> Path:
    """Create a valid YOLOv8 detection dataset structure."""
    # Create directories
    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "train" / "labels").mkdir(parents=True)
    (tmp_path / "val" / "images").mkdir(parents=True)
    (tmp_path / "val" / "labels").mkdir(parents=True)

    # Create data.yaml
    data_yaml = {
        "train": "train",
        "val": "val",
        "nc": 2,
        "names": ["hold", "volume"]
    }
    with open(tmp_path / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f)

    # Create sample images
    for i in range(5):
        (tmp_path / "train" / "images" / f"img_{i}.jpg").touch()
        (tmp_path / "val" / "images" / f"img_{i}.jpg").touch()

    return tmp_path
```

---

## 11. Implementation Checklist

### Phase 1: Package Setup

- [ ] Create `src/training/` directory
- [ ] Create `src/training/__init__.py` with exports
- [ ] Create `src/training/exceptions.py` with custom exceptions

### Phase 2: Core Implementation

- [ ] Create `src/training/datasets.py`
- [ ] Implement `validate_data_yaml()` function
- [ ] Implement `validate_directory_structure()` function
- [ ] Implement `count_dataset_images()` function
- [ ] Implement `load_hold_detection_dataset()` main function
- [ ] Add comprehensive logging

### Phase 3: Testing

- [ ] Create `tests/test_training_datasets.py`
- [ ] Implement all test fixtures
- [ ] Write all test cases from Section 10.2
- [ ] Achieve >=85% coverage for new code

### Phase 4: Documentation & QA

- [ ] Add Google-style docstrings to all functions
- [ ] Add type annotations to all functions
- [ ] Run full QA suite: mypy, ruff, pylint, pytest
- [ ] Verify pylint score >=8.5/10

---

## 12. Quality Gates

### Pre-Merge Requirements

- [ ] `mypy src/ tests/` passes with no errors
- [ ] `ruff check src/ tests/ --ignore E501` passes
- [ ] `ruff format --check src/ tests/` passes
- [ ] `pytest tests/ --cov=src --cov-fail-under=85` passes
- [ ] `pylint src/ --ignore=archive` score >= 8.5/10
- [ ] All new functions have Google-style docstrings
- [ ] All functions have complete type annotations

---

## 13. Design Decisions

### 13.1 Strict Two-Class Taxonomy

**Decision**: Enforce exactly 2 classes: hold and volume.

**Rationale**:
- Per MODEL_PRETRAIN.md: "Geometry matters more than semantic subtype"
- Reduces label noise from inconsistent subtype annotations
- Detection focuses on locating holds, not classifying them
- Classification happens separately in PR-4.x

### 13.2 Path Objects vs Strings

**Decision**: Return Path objects for directory paths.

**Rationale**:
- Type safety with pathlib.Path
- Cross-platform compatibility
- Easy manipulation (joining, checking existence)
- Consistent with Python best practices

### 13.3 Optional Test Split

**Decision**: Test split is optional.

**Rationale**:
- Many Roboflow exports only have train/val
- Test set not required for initial training
- Avoids blocking on dataset structure variations

### 13.4 Strict Mode Default

**Decision**: Default `strict=True` for validation.

**Rationale**:
- Fail early with clear error messages
- Prevent silent data quality issues
- Can be disabled for exploratory use

---

## 14. Integration Points

### 14.1 Required By

| Future PR | Usage |
|-----------|-------|
| PR-3.2 Detection Model | Uses dataset config for model setup |
| PR-3.3 Training Loop | Loads dataset for training |
| PR-3.4 Detection Inference | References class names |

### 14.2 No Dependencies

This PR has no dependencies on other PRs and can be developed in parallel with M1/M2.

---

## 15. References

- [plans/MIGRATION_PLAN.md](../MIGRATION_PLAN.md) — Migration roadmap (PR-3.1 spec)
- [docs/MODEL_PRETRAIN.md](../../docs/MODEL_PRETRAIN.md) — Detection task definition
- [docs/DESIGN.md](../../docs/DESIGN.md) — Architecture overview
- [data/sample_hold/data.yaml](../../data/sample_hold/data.yaml) — Example format (legacy 5-class)
- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/datasets/detect/)

---

## Changelog

- **2026-01-28**: Initial specification draft created
