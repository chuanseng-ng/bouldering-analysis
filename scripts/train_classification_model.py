"""Train the hold type classification model from the detection dataset.

This script has two phases:

Phase 1 — Crop extraction:
    Reads the YOLO-format detection dataset (``data/hold_classification``),
    extracts bounding-box crops for each annotated hold, and writes them into
    a folder-per-class layout required by the classification trainer.

    Class mapping from detection labels (DETECTION_CLASS_MAP):
        0 (Crimp)      → ``crimp/``
        1 (Edges)      → ``edges/``
        2 (Foothold)   → ``foothold/``
        3 (Hand-holds) → ``unknown/``
        4 (Jug)        → ``jug/``
        5 (Pinch)      → ``pinch/``
        6 (Pocket)     → ``pocket/``
        7 (Sloper)     → ``sloper/``

    The train split must contain at least one crop for every class; the
    script exits with an error if any class is missing.  Val/test splits
    may be empty (no placeholder images are copied).

Phase 2 — Classification training:
    Calls :func:`src.training.train_classification.train_hold_classifier` on
    the prepared crops dataset.

Usage:
    uv run python scripts/train_classification_model.py
    uv run python scripts/train_classification_model.py --skip-extraction
    uv run python scripts/train_classification_model.py --epochs 30 --architecture mobilenet_v3_small

Success criteria (MODEL_PRETRAIN.md §6.6):
    - Top-1 accuracy >= 0.80 on the validation set
    - ECE < 0.10
"""

import argparse
import shutil
import sys
import warnings
from pathlib import Path

import yaml

from PIL import Image  # type: ignore[import-untyped]
import torch

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

# pylint: disable=wrong-import-position,duplicate-code

from src.training.classification_dataset import (
    HOLD_CLASSES,
    load_hold_classification_dataset,
)
from src.training.datasets import EXPECTED_CLASSES
from src.training.classification_model import ClassifierHyperparameters
from src.training.train_classification import (
    ClassificationTrainingResult,
    train_hold_classifier,
)

# ── thresholds from MODEL_PRETRAIN.md §6.6 ─────────────────────────────────
ACCURACY_THRESHOLD: float = 0.80
ECE_THRESHOLD: float = 0.10

# ── crop extraction constants ────────────────────────────────────────────────
CROP_SIZE: tuple[int, int] = (224, 224)
# Maps YOLO class index (from data.yaml order) to normalised HOLD_CLASSES name.
# Dataset order: 0=Crimp, 1=Edges, 2=Foothold, 3=Hand-holds, 4=Jug, 5=Pinch, 6=Pocket, 7=Sloper
DETECTION_CLASS_MAP: dict[int, str] = {
    0: "crimp",
    1: "edges",
    2: "foothold",
    3: "unknown",  # Hand-holds → generic unknown
    4: "jug",
    5: "pinch",
    6: "pocket",
    7: "sloper",
}
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train hold-type classification model from detection dataset crops.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-dataset",
        type=Path,
        default=Path("data/hold_classification"),
        help="Root directory of the YOLO-format detection dataset to crop from.",
    )
    parser.add_argument(
        "--crops-dataset",
        type=Path,
        default=Path("data/hold_classification_crops"),
        help="Output root for the folder-per-class crops dataset.",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip crop extraction and use an existing crops dataset.",
    )
    parser.add_argument(
        "--architecture",
        choices=["resnet18", "mobilenet_v3_small", "mobilenet_v3_large"],
        default="resnet18",
        help="CNN backbone for the classifier.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory for model artifacts (default: models/classification/).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Compute device: 'cuda', 'cpu', '0', or '' for auto-detect.",
    )
    return parser.parse_args()


# ── Phase 1: crop extraction ─────────────────────────────────────────────────


def _extract_crops_for_split(  # pylint: disable=too-many-locals
    images_dir: Path,
    labels_dir: Path,
    output_split_dir: Path,
) -> dict[str, int]:
    """Extract hold crops from one dataset split into folder-per-class layout.

    Args:
        images_dir: Directory containing source images.
        labels_dir: Directory containing YOLO .txt label files (same stems).
        output_split_dir: Root of the split output (e.g. ``crops/train/``).

    Returns:
        Dict mapping class name to number of crops saved.
    """
    counts: dict[str, int] = {cls: 0 for cls in HOLD_CLASSES}

    # Create all 8 class directories up front.
    for cls in HOLD_CLASSES:
        (output_split_dir / cls).mkdir(parents=True, exist_ok=True)

    image_paths = [
        p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    for img_path in image_paths:
        label_path = labels_dir / img_path.with_suffix(".txt").name
        if not label_path.exists():
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except OSError:
            warnings.warn(f"Could not open image: {img_path}", stacklevel=2)
            continue

        img_w, img_h = image.size
        lines = [ln.strip() for ln in label_path.read_text().splitlines() if ln.strip()]

        for idx, line in enumerate(lines):
            parts = line.split()
            if len(parts) != 5:
                continue
            try:
                cls_id = int(parts[0])
                xc, yc, bw, bh = (
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                )
            except ValueError:
                continue

            cls_name = DETECTION_CLASS_MAP.get(cls_id, "unknown")

            # Convert normalised YOLO coords to pixel bounding box.
            x1 = max(0, int((xc - bw / 2) * img_w))
            y1 = max(0, int((yc - bh / 2) * img_h))
            x2 = min(img_w, int((xc + bw / 2) * img_w))
            y2 = min(img_h, int((yc + bh / 2) * img_h))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = image.crop((x1, y1, x2, y2)).resize(
                CROP_SIZE, Image.Resampling.BILINEAR
            )
            out_path = output_split_dir / cls_name / f"{img_path.stem}_{idx}.jpg"
            crop.save(out_path, quality=92)
            counts[cls_name] += 1

    return counts


def _add_placeholder_images(split_dir: Path, source_class: str) -> list[str]:
    """Copy one image from ``source_class`` into each empty class folder.

    The classifier's weight computation requires every class to have ≥1 image.
    Classes not present in the detection dataset (jug, crimp, sloper, pinch)
    receive one placeholder copy so the pipeline can run end-to-end.

    Args:
        split_dir: Split root (e.g. ``crops/train/``).
        source_class: Class folder to copy from (must be non-empty).

    Returns:
        List of class names that received a placeholder.
    """
    source_dir = split_dir / source_class
    source_images = [
        p
        for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not source_images:
        return []

    donor = source_images[0]
    patched: list[str] = []

    for cls in HOLD_CLASSES:
        cls_dir = split_dir / cls
        if not cls_dir.is_dir():
            cls_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(donor, cls_dir / f"_placeholder_{cls}.jpg")
            patched.append(cls)
            continue
        has_images = any(
            p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            for p in cls_dir.iterdir()
        )
        if not has_images:
            shutil.copy2(donor, cls_dir / f"_placeholder_{cls}.jpg")
            patched.append(cls)

    return patched


def _validate_dataset_names(source_dataset: Path) -> None:
    """Assert that data.yaml class names match the expected 8-class order.

    Loads ``data.yaml`` from ``source_dataset`` and verifies (case-insensitively)
    that the ``names`` list equals :data:`EXPECTED_CLASSES`.  Exits with code 1 if
    the file is missing or the names do not match, so DETECTION_CLASS_MAP cannot
    silently misassign labels.

    Args:
        source_dataset: Root of the YOLO-format detection dataset.
    """
    yaml_path = source_dataset / "data.yaml"
    if not yaml_path.exists():
        print(
            f"ERROR: data.yaml not found at {yaml_path}.\n"
            "  DETECTION_CLASS_MAP is keyed by index and requires data.yaml to verify "
            "class order.  Cannot proceed without it.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        with yaml_path.open() as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(
            f"ERROR: Failed to parse {yaml_path}: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    names = config.get("names", [])
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]

    if [str(n).lower() for n in names] != [e.lower() for e in EXPECTED_CLASSES]:
        print(
            f"ERROR: data.yaml class names do not match the expected order.\n"
            f"  Expected : {EXPECTED_CLASSES}\n"
            f"  Got      : {list(names)}\n"
            "  DETECTION_CLASS_MAP is keyed by index and would silently misassign "
            "labels if the order differs.  Fix data.yaml or update DETECTION_CLASS_MAP.",
            file=sys.stderr,
        )
        sys.exit(1)


def extract_crops(source_dataset: Path, crops_dataset: Path) -> None:
    """Run Phase 1: extract crops from every split of the detection dataset.

    Args:
        source_dataset: Root of the YOLO-format detection dataset.
        crops_dataset: Root where the folder-per-class crops will be written.
    """
    print(f"\nPhase 1 — Extracting crops from: {source_dataset}")
    print(f"           Output directory      : {crops_dataset}")
    print(
        "\n  [i]  Dataset has 8 fine-grained classes."
        "\n     Crops are labelled directly from detection annotations:"
        "\n     crimp / edges / foothold / unknown / jug / pinch / pocket / sloper."
    )

    _validate_dataset_names(source_dataset)

    for split in ("train", "val", "test"):
        images_dir = source_dataset / split / "images"
        labels_dir = source_dataset / split / "labels"

        if not images_dir.is_dir():
            print(f"  Skipping split '{split}' (no images/ directory).")
            continue

        output_split_dir = crops_dataset / split
        if output_split_dir.exists() and output_split_dir.is_dir():
            if output_split_dir.resolve() != (source_dataset / split).resolve():
                shutil.rmtree(output_split_dir)
        output_split_dir.mkdir(parents=True, exist_ok=True)
        counts = _extract_crops_for_split(images_dir, labels_dir, output_split_dir)

        total = sum(counts.values())
        print(f"\n  Split '{split}': {total} crops extracted.")
        for cls in HOLD_CLASSES:
            print(f"    {cls:<10}: {counts.get(cls, 0)}")

        # Train split must have at least one crop per class so
        # compute_class_weights() does not fail.  Val/test may be empty.
        if split == "train":
            missing = [cls for cls in HOLD_CLASSES if counts.get(cls, 0) == 0]
            if missing:
                print(
                    f"\nERROR: Train split is missing crops for class(es): {missing}.\n"
                    "  Every class must have at least one training image.\n"
                    "  Add annotated images for the missing classes and re-run.",
                    file=sys.stderr,
                )
                sys.exit(1)

    print("\nCrop extraction complete.")


# ── Phase 2: training ────────────────────────────────────────────────────────


def _print_result(result: ClassificationTrainingResult) -> None:
    """Print a formatted training result summary.

    Args:
        result: Completed classification training result.
    """
    m = result.metrics
    acc_ok = m.top1_accuracy >= ACCURACY_THRESHOLD
    ece_ok = m.ece < ECE_THRESHOLD

    print("\n" + "=" * 60)
    print("CLASSIFICATION TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Version      : {result.version}")
    print(f"  Architecture : {result.architecture}")
    print(f"  Trained at   : {result.trained_at}")
    print(f"  Best epoch   : {m.best_epoch}")
    print()
    print("  Metrics:")
    print(
        f"    Top-1 Acc   : {m.top1_accuracy:.4f}  {'[OK]' if acc_ok else '[!!]'}  (threshold: {ACCURACY_THRESHOLD})"
    )
    print(
        f"    ECE         : {m.ece:.4f}  {'[OK]' if ece_ok else '[!!]'}  (threshold: < {ECE_THRESHOLD})"
    )
    print(f"    Val loss    : {m.val_loss:.4f}")
    print()
    print("  Artifacts:")
    print(f"    Best weights : {result.best_weights_path}")
    print(f"    Metadata     : {result.metadata_path}")
    print("=" * 60)

    if not acc_ok:
        print(
            f"\n[!]  Accuracy {m.top1_accuracy:.4f} is below the target {ACCURACY_THRESHOLD}."
            "\n   Consider: more epochs, larger architecture, or additional annotated data."
        )
    elif not ece_ok:
        print(
            f"\n[!]  ECE {m.ece:.4f} exceeds the target {ECE_THRESHOLD}."
            "\n   Consider calibration techniques or additional training data."
        )
    else:
        print("\n[OK]  Accuracy target met.")
        print("[OK]  ECE target met.")


def main() -> int:
    """Entry point for the classification training script.

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    args = _parse_args()

    # ── Phase 1: extract crops ─────────────────────────────────────────────
    if args.skip_extraction:
        print(
            f"Skipping crop extraction — using existing dataset at: {args.crops_dataset}"
        )
        if not args.crops_dataset.is_dir():
            print(
                f"ERROR: Crops dataset not found at {args.crops_dataset}",
                file=sys.stderr,
            )
            return 1
    else:
        if not args.source_dataset.is_dir():
            print(
                f"ERROR: Source dataset not found at {args.source_dataset}",
                file=sys.stderr,
            )
            return 1
        extract_crops(args.source_dataset, args.crops_dataset)

    # ── Phase 2: load dataset + train ─────────────────────────────────────
    print(f"\nPhase 2 — Loading crops dataset from: {args.crops_dataset}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        dataset = load_hold_classification_dataset(args.crops_dataset, strict=False)

    print(
        f"  Classes : {dataset['names']} ({dataset['nc']} total)\n"
        f"  Train   : {dataset['train_image_count']} images\n"
        f"  Val     : {dataset['val_image_count']} images"
    )
    print("  Per-class counts (train):")
    for cls, cnt in dataset["class_counts"].items():
        print(f"    {cls:<10}: {cnt}")

    hyperparameters = ClassifierHyperparameters(
        architecture=args.architecture,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )

    _actual = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    _device_label = args.device if args.device else f"auto ({_actual})"

    print(
        f"\nTraining config:"
        f"\n  Architecture : {hyperparameters.architecture}"
        f"\n  Epochs       : {hyperparameters.epochs}"
        f"\n  Batch size   : {hyperparameters.batch_size}"
        f"\n  LR           : {hyperparameters.learning_rate}"
        f"\n  Device       : {_device_label}"
    )

    print("\nStarting training ...\n")
    result = train_hold_classifier(
        dataset=dataset,
        dataset_root=args.crops_dataset,
        hyperparameters=hyperparameters,
        output_dir=args.output_dir,
    )

    _print_result(result)
    success = (
        result.metrics.top1_accuracy >= ACCURACY_THRESHOLD
        and result.metrics.ece < ECE_THRESHOLD
    )
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
