"""Train the hold detection model on the climbing holds dataset.

Usage:
    uv run python scripts/train_detection_model.py
    uv run python scripts/train_detection_model.py --epochs 50 --model-size yolov8s
    uv run python scripts/train_detection_model.py --dataset data/hold_classification --batch-size 8

The script loads the YOLO-format dataset, trains a YOLOv8 detection model,
and saves versioned artifacts under ``models/detection/``.

Success criteria (MODEL_PRETRAIN.md §5.5):
    - Recall >= 0.85 on validation set
    - mAP50 used as secondary metric
"""

import argparse
import sys
from pathlib import Path

import torch

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

# pylint: disable=wrong-import-position,duplicate-code
from src.training.datasets import load_hold_detection_dataset
from src.training.detection_model import (
    DEFAULT_MODEL_SIZE,
    VALID_MODEL_SIZES,
    DetectionHyperparameters,
)
from src.training.train_detection import TrainingResult, train_hold_detector

# ── thresholds from MODEL_PRETRAIN.md §5.5 ─────────────────────────────────
RECALL_THRESHOLD: float = 0.85
MAP50_THRESHOLD: float = 0.80


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 hold detection model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/hold_classification"),
        help="Root directory of YOLO-format detection dataset (must contain data.yaml).",
    )
    parser.add_argument(
        "--model-size",
        choices=VALID_MODEL_SIZES,
        default=DEFAULT_MODEL_SIZE,
        help="YOLOv8 model variant.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size (-1 for auto-batch).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early-stopping patience (epochs without improvement).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory for model artifacts (default: models/detection/).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="Compute device: 'cuda', 'cpu', '0', or '' for auto-detect.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Cap number of training images (random sample, seed=42). Default: no cap.",
    )
    parser.add_argument(
        "--finetune-from",
        type=Path,
        default=None,
        help="Path to weights/best.pt to warm-start from (YOLOv8 native fine-tuning).",
    )
    return parser.parse_args()


def _print_result(result: TrainingResult) -> None:
    """Print a formatted summary of the training result.

    Args:
        result: Completed training result with metrics and artifact paths.
    """
    m = result.metrics
    recall_ok = m.recall >= RECALL_THRESHOLD
    map50_ok = m.map50 >= MAP50_THRESHOLD

    print("\n" + "=" * 60)
    print("DETECTION TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Version      : {result.version}")
    print(f"  Model size   : {result.model_size}")
    print(f"  Trained at   : {result.trained_at}")
    print(f"  Best epoch   : {m.best_epoch}")
    print()
    print("  Metrics:")
    print(
        f"    Recall      : {m.recall:.4f}  {'[OK]' if recall_ok else '[!!]'}  (threshold: {RECALL_THRESHOLD})"
    )
    print(
        f"    mAP50       : {m.map50:.4f}  {'[OK]' if map50_ok else '[!!]'}  (threshold: {MAP50_THRESHOLD})"
    )
    print(f"    mAP50-95    : {m.map50_95:.4f}")
    print(f"    Precision   : {m.precision:.4f}")
    print()
    print("  Artifacts:")
    print(f"    Best weights : {result.best_weights_path}")
    print(f"    Metadata     : {result.metadata_path}")
    print("=" * 60)

    if not recall_ok:
        print(
            f"\n[!]  Recall {m.recall:.4f} is below the target {RECALL_THRESHOLD}."
            "\n   Consider: more epochs, larger model, or additional data."
        )
    else:
        print("\n[OK]  Recall target met. Model is ready for inference.")


def main() -> int:
    """Entry point for the detection training script.

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    args = _parse_args()

    # ── validate dataset ───────────────────────────────────────────────────
    print(f"Loading detection dataset from: {args.dataset}")
    dataset = load_hold_detection_dataset(args.dataset, max_images=args.max_images)

    print(
        f"  Classes : {dataset['names']} ({dataset['nc']} total)\n"
        f"  Train   : {dataset['train_image_count']} images\n"
        f"  Val     : {dataset['val_image_count']} images\n"
        f"  Test    : {dataset['test_image_count']} images"
    )

    # ── configure hyperparameters ──────────────────────────────────────────
    hyperparameters = DetectionHyperparameters(
        epochs=args.epochs,
        batch=args.batch_size,
        patience=args.patience,
        device=args.device,
    )

    _actual = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    _device_label = args.device if args.device else f"auto ({_actual})"

    _max_images_label = str(args.max_images) if args.max_images is not None else "none"
    _finetune_label = (
        str(args.finetune_from) if args.finetune_from is not None else "none"
    )

    print(
        f"\nTraining config:"
        f"\n  Model size   : {args.model_size}"
        f"\n  Epochs       : {hyperparameters.epochs}"
        f"\n  Batch size   : {hyperparameters.batch_size}"
        f"\n  Patience     : {hyperparameters.patience}"
        f"\n  Device       : {_device_label}"
        f"\n  Max images   : {_max_images_label}"
        f"\n  Finetune from: {_finetune_label}"
    )

    # ── train ──────────────────────────────────────────────────────────────
    print("\nStarting training ...\n")
    result = train_hold_detector(
        dataset=dataset,
        dataset_root=args.dataset,
        hyperparameters=hyperparameters,
        output_dir=args.output_dir,
        model_size=args.model_size,
        finetune_from=args.finetune_from,
        max_images=args.max_images,
    )

    _print_result(result)
    return 0 if result.metrics.recall >= RECALL_THRESHOLD else 1


if __name__ == "__main__":
    sys.exit(main())
