"""Verify the detection model and determine whether re-training is needed.

The script searches for trained detection models in order:
    1. ``models/detection/v*/weights/best.pt``  (produced by train_detection_model.py)
    2. ``models/hold_detection/*.pt``            (pre-existing weights)

It then runs YOLO validation on the supplied dataset and reports whether the
model meets the recall threshold from MODEL_PRETRAIN.md §5.5.

Usage:
    uv run python scripts/verify_detection_model.py
    uv run python scripts/verify_detection_model.py --weights models/hold_detection/test_v1.pt
    uv run python scripts/verify_detection_model.py --metadata-only
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

# pylint: disable=wrong-import-position,duplicate-code

# ── thresholds from MODEL_PRETRAIN.md §5.5 ─────────────────────────────────
RECALL_THRESHOLD: float = 0.85
MAP50_THRESHOLD: float = 0.80

# ── candidate model locations ───────────────────────────────────────────────
TRAINED_MODELS_DIR = Path("models/detection")
LEGACY_MODELS_DIR = Path("models/hold_detection")


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace.
    """
    parser = argparse.ArgumentParser(
        description="Verify hold detection model quality and decide if re-training is needed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to model weights (.pt). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/hold_classification"),
        help="Root directory of YOLO-format dataset (must contain data.yaml).",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only read stored metadata; do not run YOLO validation.",
    )
    return parser.parse_args()


def _find_latest_trained_weights() -> Path | None:
    """Search ``models/detection/`` for the most recent versioned best.pt.

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


def _find_legacy_weights() -> Path | None:
    """Search ``models/hold_detection/`` for any pre-existing .pt file.

    Prefers ``new_v1.pt`` > ``test_v1.pt`` > any other .pt.

    Returns:
        Path to a pre-existing weights file, or None.
    """
    if not LEGACY_MODELS_DIR.is_dir():
        return None

    preferred = ["new_v1.pt", "test_v1.pt"]
    for name in preferred:
        p = LEGACY_MODELS_DIR / name
        if p.exists():
            return p

    # Fall back to any .pt file — sort for deterministic selection.
    pts = sorted(LEGACY_MODELS_DIR.glob("*.pt"), key=lambda p: p.name)
    return pts[0] if pts else None


def _read_trained_metadata(weights_path: Path) -> dict[str, Any] | None:
    """Try to read metadata.json adjacent to a versioned best.pt.

    Args:
        weights_path: Path to best.pt inside a version directory.

    Returns:
        Parsed metadata dict, or None if not found.
    """
    metadata_path = weights_path.parent.parent / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        result: dict[str, Any] = json.loads(metadata_path.read_text())
        return result
    except (json.JSONDecodeError, OSError):
        return None


def _read_legacy_metadata(weights_path: Path) -> dict[str, Any] | None:
    """Try to read a .yaml metadata file next to a legacy .pt file.

    Args:
        weights_path: Path to a legacy weights file.

    Returns:
        Parsed metadata dict (via PyYAML), or None if not found.
    """
    import yaml  # type: ignore[import-untyped]  # pylint: disable=import-outside-toplevel

    yaml_path = weights_path.with_name(weights_path.stem + "_metadata.yaml")
    if not yaml_path.exists():
        return None
    try:
        with yaml_path.open() as fh:
            data: dict[str, Any] = yaml.safe_load(fh)
            return data
    except (yaml.YAMLError, OSError):
        return None


def _print_metadata_summary(metadata: dict[str, Any], source: str) -> None:
    """Print a human-readable summary of the stored metadata.

    Args:
        metadata: Parsed metadata dictionary.
        source: Label for the metadata source (e.g. ``"metadata.json"``).
    """
    print(f"\n  Stored metrics ({source}):")
    metrics = metadata.get("metrics", metadata)  # support both layouts
    for key, val in metrics.items():
        print(f"    {key}: {val}")


def _run_yolo_validation(weights_path: Path, data_yaml: Path) -> dict[str, Any]:
    """Run YOLO validation and return a metrics dict.

    Args:
        weights_path: Path to model weights (.pt).
        data_yaml: Path to the dataset data.yaml file.

    Returns:
        Dict with keys 'recall', 'map50', 'map50_95', 'precision'.

    Raises:
        ImportError: If ultralytics is not installed.
        RuntimeError: If validation fails.
    """
    from ultralytics import YOLO  # type: ignore[import-untyped]  # pylint: disable=import-outside-toplevel

    print("\n  Running YOLO validation ...")
    model = YOLO(str(weights_path))
    results = model.val(data=str(data_yaml), verbose=False)

    rd = results.results_dict
    return {
        "recall": rd.get("metrics/recall(B)", 0.0),
        "map50": rd.get("metrics/mAP50(B)", 0.0),
        "map50_95": rd.get("metrics/mAP50-95(B)", 0.0),
        "precision": rd.get("metrics/precision(B)", 0.0),
    }


def _print_verdict(recall: float, map50: float) -> bool:
    """Print pass/fail verdict and re-training recommendation.

    Args:
        recall: Validation recall value.
        map50: Validation mAP50 value.

    Returns:
        True if the model meets all thresholds, False otherwise.
    """
    recall_ok = recall >= RECALL_THRESHOLD
    map50_ok = map50 >= MAP50_THRESHOLD
    passed = recall_ok and map50_ok

    print("\n" + "=" * 60)
    print("DETECTION MODEL VERDICT")
    print("=" * 60)
    print(
        f"  Recall  : {recall:.4f}  {'[PASS]' if recall_ok else '[FAIL]'}  (threshold: {RECALL_THRESHOLD})"
    )
    print(
        f"  mAP50   : {map50:.4f}  {'[PASS]' if map50_ok else '[FAIL]'}  (threshold: {MAP50_THRESHOLD})"
    )
    print()

    if passed:
        print("  [OK]  Model meets all thresholds. Re-training NOT needed.")
    else:
        print("  [!!]  Model does not meet thresholds. Re-training RECOMMENDED.")
        if not recall_ok:
            print(
                f"     → Recall {recall:.4f} < {RECALL_THRESHOLD}: focus on recall (lower conf threshold or more data)."
            )
        if not map50_ok:
            print(
                f"     → mAP50 {map50:.4f} < {MAP50_THRESHOLD}: improve precision-recall balance."
            )
        print("\n  To re-train:")
        print("     uv run python scripts/train_detection_model.py")
    print("=" * 60)
    return passed


def main() -> int:
    """Entry point for the detection verification script.

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
        is_legacy = "hold_detection" in str(weights_path)
    else:
        weights_path = _find_latest_trained_weights()
        is_legacy = False
        if weights_path is None:
            weights_path = _find_legacy_weights()
            is_legacy = True
        if weights_path is None:
            print(
                "ERROR: No detection model found.\n"
                f"  Searched: {TRAINED_MODELS_DIR}/  and  {LEGACY_MODELS_DIR}/\n"
                "  Run: uv run python scripts/train_detection_model.py",
                file=sys.stderr,
            )
            return 1
        print(f"Auto-detected weights: {weights_path}")

    # ── read stored metadata ───────────────────────────────────────────────
    metadata = (
        _read_legacy_metadata(weights_path)
        if is_legacy
        else _read_trained_metadata(weights_path)
    )
    if metadata:
        _print_metadata_summary(
            metadata, "metadata.yaml" if is_legacy else "metadata.json"
        )
    else:
        print("  (No stored metadata found — skipping cached metrics.)")

    # ── run live validation (unless --metadata-only) ───────────────────────
    if args.metadata_only:
        print("\n--metadata-only: skipping live validation.")
        if metadata:
            # Extract recall from stored metadata for the verdict.
            metrics = metadata.get("metrics", metadata)
            try:
                recall = float(metrics.get("recall", 0.0))
                map50 = float(metrics.get("map50", metrics.get("final_mAP50", 0.0)))
            except (TypeError, ValueError):
                print(
                    "WARNING: Invalid metric values in metadata; cannot determine verdict.",
                    file=sys.stderr,
                )
                return 1
            passed = _print_verdict(recall, map50)
            return 0 if passed else 1
        print(
            "WARNING: No metadata and --metadata-only set; cannot determine verdict.",
            file=sys.stderr,
        )
        return 1

    data_yaml = args.dataset / "data.yaml"
    if not data_yaml.exists():
        print(f"ERROR: data.yaml not found at {data_yaml}", file=sys.stderr)
        return 1

    try:
        live = _run_yolo_validation(weights_path, data_yaml)
    except Exception as exc:
        print(f"ERROR during validation: {exc}", file=sys.stderr)
        return 1

    print(
        f"\n  Live validation results:"
        f"\n    Recall    : {live['recall']:.4f}"
        f"\n    mAP50     : {live['map50']:.4f}"
        f"\n    mAP50-95  : {live['map50_95']:.4f}"
        f"\n    Precision : {live['precision']:.4f}"
    )

    passed = _print_verdict(live["recall"], live["map50"])
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
