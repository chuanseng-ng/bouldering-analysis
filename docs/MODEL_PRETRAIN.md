# Model Pretraining Specification

## Bouldering Route Analysis — Perception Models

---

## 1. Purpose & Scope

This document defines the **model pretraining pipeline** for the bouldering route
analysis system.

The goal of pretraining is to produce **reusable, gym-agnostic perception models**
that extract structured information from route images:

- Hold & volume detection
- Hold type classification

These models are trained **offline**, versioned, and **frozen for online inference**.

⚠️ This spec explicitly excludes:

- Route grading
- Feature extraction
- Graph modeling
- Frontend concerns

---

## 2. Design Principles

1. Pretrain perception, not difficulty
2. Separate training code from inference code
3. Optimize recall over precision for detection
4. Keep label taxonomies minimal
5. Every model output must include confidence
6. Models must be replaceable without API changes

---

## 3. Tasks Overview

| Task | Type | Output |
| :--: | :--: | :----: |
| Hold Detection | Object Detection | Bounding boxes / masks |
| Hold Classification | Image Classification | Hold type probabilities |

---

## 4. Datasets

### 4.1 Primary Source

- Roboflow-hosted climbing hold / volume datasets
- Export formats:
  - YOLOv8 (detection)
  - Image classification (cropped holds)

### 4.2 Dataset Versioning

Each dataset version MUST include:

- Source name
- Export date
- Augmentation config
- Class taxonomy

Example:

```yaml
dataset_version: rf-climbing-holds-v1
export_format: yolov8
classes: [hold, volume]
```

## 5. Hold & Volume Detection Pretraining

### 5.1 Task Definition

- Input
  - Full climbing wall image
- Output
  - Bounding boxes (normalized)
  - Class: hold or volume
  - Confidence score

### 5.2 Class Taxonomy (Detection)

```text
0: hold
1: volume
```

- Rationale:
  - Geometry matters more than semantic subtype
  - Reduces label noise
  - Improves recall

### 5.3 Model Architecture

- Baseline
  - YOLOv8m (recommended)
  - Input resolution: 640×640
- Alternatives
  - YOLOv8l (higher accuracy)
  - Detectron2 (if segmentation needed)

### 5.4 Training Configuration

- Loss: default YOLO detection loss
- Optimizer: AdamW
- Epochs: 50–100
- Augmentations:
  - Random rotation
  - Brightness / contrast
  - Gaussian noise
  - Perspective warp (mild)

### 5.5 Evaluation Metrics

- Primary:
  - Recall @ IoU 0.5
- Secondary:
  - Precision
  - mAP50
- Success criteria:
  - Recall ≥ 0.85 on validation set

### 5.6 Output Artifacts

Saved to:

```bash
models/detection/
  ├── weights.pt
  ├── classes.yaml
  └── metadata.json
```

metadata.json includes:

- training dataset version
- commit hash
- training date
- input resolution

### 5.7 Inference Contract

```python
def detect_holds(image: np.ndarray) -> list[DetectedHold]:
    """
    Returns:
      - bounding box (x, y, w, h) normalized
      - class (hold | volume)
      - confidence
    """
```

## 6. Hold Type Classification Pretraining

### 6.1 Task Definition

- Input
  - Cropped image of a detected hold
- Output
  - Probability distribution over hold types

### 6.2 Class Taxonomy (Classification)

MVP taxonomy:

```text
jug
crimp
sloper
pinch
volume
unknown
```

- Notes:
  - unknown absorbs ambiguity
  - Confidence is mandatory
  - Taxonomy may expand later

### 6.3 Dataset Construction

- Two supported sources:
  - Roboflow-labeled hold types (if available)
  - Manually curated subset (high-quality labels)

- Minimum recommended size:
  - 1,000–3,000 images total

### 6.4 Model Architecture

- Baseline
  - ResNet-18 or MobileNetV3
  - Input: 224×224 RGB
- Training Heads
  - Softmax output
  - Label smoothing enabled

### 6.5 Training Configuration

- Loss: Cross-entropy
- Optimizer: Adam
- Augmentations:
  - Random rotation
  - Color jitter
  - Cutout
- Class imbalance handling:
  - Weighted loss or oversampling

### 6.6 Evaluation Metrics

- Primary:
  - Top-1 accuracy
- Secondary:
  - Confusion matrix
  - Calibration error (ECE)
- Success criteria:
  - ≥70% Top-1 accuracy
  - Reasonable calibration (ECE < 0.1)

### 6.7 Output Artifacts

Saved to:

```text
models/classification/
  ├── weights.pt
  ├── classes.json
  └── metadata.json
```

### 6.8 Inference Contract

```python
def classify_hold(crop: np.ndarray) -> HoldTypeResult:
    """
    Returns:
      - predicted type
      - probability distribution
      - confidence score
    """
```

## 7. Model Freezing & Deployment Rules

- Pretrained models are read-only in production
- No training code in API container
- Model updates require:
- New version folder
- Explicit backend config change

## 8. Integration Guarantees

- The perception layer guarantees:
  - Deterministic output for identical inputs
  - Stable output schema across versions
  - Confidence scores on all predictions
- Downstream systems MUST:
  - Treat outputs probabilistically
  - Never assume perfect classification

## 9. Failure Modes & Mitigations

| Issue | Mitigation |
| :---: | :--------: |
| Missed holds | Optimize recall |
| Misclassified slopers | Use confidence thresholds |
| Gym lighting changes | Strong augmentation |
| New hold styles | Fine-tune later |

## 10. Non-Goals

- Grading prediction
- Route sequencing
- Beta inference
- Gym-specific tuning

## 11. Future Extensions

- Instance segmentation for hold shape
- Multi-label classification (hand/foot usability)
- Gym-adaptive fine-tuning
- Vision-language hold descriptors

## 12. Summary

- This pretraining pipeline:
  - Decouples perception from grading
  - Leverages existing datasets efficiently
  - Produces reusable, explainable models
  - Enables rapid iteration downstream
  - Pretrain perception once. Build intelligence on top.
