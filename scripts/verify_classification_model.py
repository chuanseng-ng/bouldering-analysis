"""Verify the classification model and determine whether re-training is needed.

The script searches for trained classification models under
``models/classification/v*/weights/best.pt`` (produced by
train_classification_model.py), loads the model, and runs inference on the
validation split to compute top-1 accuracy and ECE.

Usage:
    uv run python scripts/verify_classification_model.py
    uv run python scripts/verify_classification_model.py --weights models/classification/v.../weights/best.pt
    uv run python scripts/verify_classification_model.py --dataset data/hold_classification_crops
    uv run python scripts/verify_classification_model.py --metadata-only

Success criteria (MODEL_PRETRAIN.md §6.6):
    - Top-1 accuracy >= 0.80
    - ECE < 0.10
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

# pylint: disable=wrong-import-position,duplicate-code

# ── thresholds from MODEL_PRETRAIN.md §6.6 ─────────────────────────────────
ACCURACY_THRESHOLD: float = 0.80
ECE_THRESHOLD: float = 0.10

TRAINED_MODELS_DIR = Path("models/classification")
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png"})
ECE_BINS: int = 10


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Verify hold classification model quality and decide if re-training is needed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to classification weights (.pt). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/hold_classification_crops"),
        help="Root of the folder-per-class validation dataset.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only read stored metadata; do not run inference.",
    )
    return parser.parse_args()


def _find_latest_trained_weights() -> Path | None:
    """Search ``models/classification/`` for the most recent versioned best.pt.

    Returns:
        Path to best.pt of the latest version, or None if no versions exist.
    """
    if not TRAINED_MODELS_DIR.is_dir():
        return None

    versions = sorted(
        [
            d
            for d in TRAINED_MODELS_DIR.iterdir()
            if d.is_dir() and d.name.startswith("v")
        ],
        key=lambda d: d.name,
        reverse=True,
    )
    for version_dir in versions:
        weights = version_dir / "weights" / "best.pt"
        if weights.exists():
            return weights
    return None


def _read_metadata(weights_path: Path) -> dict[str, Any] | None:
    """Read metadata.json for a versioned classification model.

    Args:
        weights_path: Path to best.pt inside a version directory.

    Returns:
        Parsed metadata dict, or None if not found / unreadable.
    """
    metadata_path = weights_path.parent.parent / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        result: dict[str, Any] = json.loads(metadata_path.read_text())
        return result
    except (json.JSONDecodeError, OSError):
        return None


def _compute_ece(
    confidences: list[float], correctness: list[bool], n_bins: int = ECE_BINS
) -> float:
    """Compute Expected Calibration Error over equally-spaced confidence bins.

    Args:
        confidences: Per-sample max predicted probability.
        correctness: Per-sample boolean (True = prediction matched label).
        n_bins: Number of confidence bins.

    Returns:
        ECE value in [0, 1].
    """
    if not confidences:
        return 0.0

    bin_size = 1.0 / n_bins
    ece = 0.0
    n = len(confidences)

    for b in range(n_bins):
        lo = b * bin_size
        hi = lo + bin_size
        if b == 0:
            indices = [i for i, c in enumerate(confidences) if lo <= c <= hi]
        else:
            indices = [i for i, c in enumerate(confidences) if lo < c <= hi]
        if not indices:
            continue
        bin_acc = sum(1 for i in indices if correctness[i]) / len(indices)
        bin_conf = sum(confidences[i] for i in indices) / len(indices)
        ece += (len(indices) / n) * abs(bin_acc - bin_conf)

    return ece


def _run_inference(  # pylint: disable=too-many-locals,too-many-statements
    weights_path: Path, val_dir: Path
) -> tuple[float, float, int]:
    """Run classification inference on all images in the val split.

    Args:
        weights_path: Path to trained classification weights (.pt).
        val_dir: Path to val/ split directory (folder-per-class layout).

    Returns:
        Tuple of (top1_accuracy, ece, total_samples).

    Raises:
        RuntimeError: If no images are found or model loading fails.
    """
    import torch  # pylint: disable=import-outside-toplevel
    from PIL import Image  # type: ignore[import-untyped]  # pylint: disable=import-outside-toplevel
    from torchvision import transforms  # type: ignore[import-untyped]  # pylint: disable=import-outside-toplevel

    from src.training.classification_model import (  # pylint: disable=import-outside-toplevel
        IMAGENET_MEAN,
        IMAGENET_STD,
        ClassifierHyperparameters,
        build_hold_classifier,
    )

    # ── load metadata to reconstruct architecture ──────────────────────────
    metadata = _read_metadata(weights_path)
    arch = "resnet18"
    input_size = 224
    if metadata:
        arch = metadata.get("hyperparameters", {}).get("architecture", arch)
        input_size = metadata.get("hyperparameters", {}).get("input_size", input_size)

    # ── build model + load weights ─────────────────────────────────────────
    print(f"\n  Loading model: architecture={arch}, input_size={input_size}")
    hyp = ClassifierHyperparameters(architecture=arch, input_size=input_size)
    config = build_hold_classifier(hyp)
    model: torch.nn.Module = config["model"]

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    # ── validation transform (matches training) ────────────────────────────
    from src.training.classification_model import VAL_RESIZE_RATIO  # pylint: disable=import-outside-toplevel

    resize_to = int(input_size * VAL_RESIZE_RATIO)
    val_transform = transforms.Compose(
        [
            transforms.Resize(resize_to),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    # ── collect val images ─────────────────────────────────────────────────
    class_dirs = [d for d in val_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        raise RuntimeError(f"No class subdirectories found in val dir: {val_dir}")

    class_names = sorted(d.name for d in class_dirs)
    class_to_idx: dict[str, int] = {name: i for i, name in enumerate(class_names)}

    print(f"  Classes found in val: {class_names}")

    correct = 0
    total = 0
    confidences: list[float] = []
    correctness: list[bool] = []

    import torch.nn.functional as F  # noqa: N812  # pylint: disable=import-outside-toplevel

    with torch.no_grad():
        for cls_dir in class_dirs:
            label = class_to_idx[cls_dir.name]
            for img_path in cls_dir.iterdir():
                if not (
                    img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS
                ):
                    continue
                try:
                    img = Image.open(img_path).convert("RGB")
                except OSError:
                    continue

                tensor = val_transform(img).unsqueeze(0)
                logits = model(tensor)
                probs = F.softmax(logits, dim=1).squeeze(0)
                pred = int(probs.argmax().item())
                conf = float(probs.max().item())

                is_correct = pred == label
                correct += int(is_correct)
                total += 1
                confidences.append(conf)
                correctness.append(is_correct)

    if total == 0:
        raise RuntimeError(f"No images found in val dir: {val_dir}")

    accuracy = correct / total
    ece = _compute_ece(confidences, correctness)
    return accuracy, ece, total


def _print_verdict(accuracy: float, ece: float) -> bool:
    """Print pass/fail verdict and re-training recommendation.

    Args:
        accuracy: Top-1 accuracy on the val set.
        ece: Expected Calibration Error on the val set.

    Returns:
        True if the model meets all thresholds, False otherwise.
    """
    acc_ok = accuracy >= ACCURACY_THRESHOLD
    ece_ok = ece < ECE_THRESHOLD
    passed = acc_ok and ece_ok

    print("\n" + "=" * 60)
    print("CLASSIFICATION MODEL VERDICT")
    print("=" * 60)
    print(
        f"  Top-1 Acc : {accuracy:.4f}  {'[PASS]' if acc_ok else '[FAIL]'}  (threshold: {ACCURACY_THRESHOLD})"
    )
    print(
        f"  ECE       : {ece:.4f}  {'[PASS]' if ece_ok else '[FAIL]'}  (threshold: < {ECE_THRESHOLD})"
    )
    print()

    if passed:
        print("  [OK]  Model meets all thresholds. Re-training NOT needed.")
    else:
        print("  [!!]  Model does not meet thresholds. Re-training RECOMMENDED.")
        if not acc_ok:
            print(
                f"     → Accuracy {accuracy:.4f} < {ACCURACY_THRESHOLD}."
                "\n       If trained on 2-class crop data, supply a 6-class Roboflow"
                "\n       hold-type dataset and re-run train_classification_model.py."
            )
        if not ece_ok:
            print(
                f"     → ECE {ece:.4f} >= {ECE_THRESHOLD}."
                "\n       Enable label_smoothing or temperature scaling."
            )
        print("\n  To re-train:")
        print("     uv run python scripts/train_classification_model.py")
    print("=" * 60)
    return passed


def main() -> int:  # pylint: disable=too-many-return-statements
    """Entry point for the classification verification script.

    Returns:
        Exit code (0 = model passes, 1 = re-training needed or error).
    """
    args = _parse_args()

    # ── locate weights ─────────────────────────────────────────────────────
    if args.weights:
        weights_path = args.weights
        if not weights_path.exists():
            print(f"ERROR: Weights file not found: {weights_path}", file=sys.stderr)
            return 1
        print(f"Using specified weights: {weights_path}")
    else:
        weights_path = _find_latest_trained_weights()
        if weights_path is None:
            print(
                "ERROR: No trained classification model found.\n"
                f"  Searched: {TRAINED_MODELS_DIR}/\n"
                "  Run: uv run python scripts/train_classification_model.py",
                file=sys.stderr,
            )
            return 1
        print(f"Auto-detected weights: {weights_path}")

    # ── read stored metadata ───────────────────────────────────────────────
    metadata = _read_metadata(weights_path)
    if metadata:
        m = metadata.get("metrics", {})
        print("\n  Stored metrics (metadata.json):")
        print(f"    Top-1 accuracy : {m.get('top1_accuracy', 'n/a')}")
        print(f"    Val loss       : {m.get('val_loss', 'n/a')}")
        print(f"    ECE            : {m.get('ece', 'n/a')}")
        print(f"    Best epoch     : {m.get('best_epoch', 'n/a')}")
    else:
        print("  (No stored metadata found.)")

    # ── metadata-only mode ─────────────────────────────────────────────────
    if args.metadata_only:
        print("\n--metadata-only: skipping live inference.")
        if metadata:
            m = metadata.get("metrics", {})
            accuracy = float(m.get("top1_accuracy") or 0.0)
            ece = float(m.get("ece") or 1.0)
            passed = _print_verdict(accuracy, ece)
            return 0 if passed else 1
        print(
            "WARNING: No metadata and --metadata-only set; cannot determine verdict.",
            file=sys.stderr,
        )
        return 1

    # ── live inference ─────────────────────────────────────────────────────
    val_dir = args.dataset / "val"
    if not val_dir.is_dir():
        print(
            f"ERROR: val/ directory not found at {val_dir}\n"
            f"  Supply --dataset pointing to a folder-per-class crops dataset.\n"
            f"  If crops have not been extracted yet, run:\n"
            f"    uv run python scripts/train_classification_model.py",
            file=sys.stderr,
        )
        return 1

    try:
        accuracy, ece, total = _run_inference(weights_path, val_dir)
    except RuntimeError as exc:
        print(f"ERROR during inference: {exc}", file=sys.stderr)
        return 1

    print(f"\n  Live inference on {total} val images:")
    print(f"    Top-1 accuracy : {accuracy:.4f}")
    print(f"    ECE            : {ece:.4f}")

    passed = _print_verdict(accuracy, ece)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
