# YOLO Model Training Guide - Current Codebase

## Executive Summary

This guide walks through how to train the YOLOv8 hold detection model using the current codebase, and explains what would be needed to add **slant angle detection** (hold orientation).

**Current Model Output**: Bounding box, hold type (8 classes), confidence  
**Missing for Slant Detection**: Hold orientation angle (not currently captured)

---

## Current Training Setup

### Overview

The codebase uses **YOLOv8n** (nano) for hold detection with the following capabilities:

**Current Classes** (8 hold types):
```python
0: crimp
1: jug
2: sloper
3: pinch
4: pocket
5: foot-hold
6: start-hold
7: top-out-hold
```

**Current Model Output** (per detection):
- Bounding box: (x1, y1, x2, y2)
- Class ID: 0-7
- Confidence: 0.0-1.0

**NOT Included**:
- Hold orientation/slant angle
- Hold rotation
- 3D information

---

## Step-by-Step: Training with Current Setup

### 1. Prepare Your Dataset

#### Required Directory Structure

```
data/your_dataset/
├── data.yaml                # Dataset configuration
├── train/
│   ├── images/              # Training images (.jpg, .png)
│   │   ├── route_001.jpg
│   │   ├── route_002.jpg
│   │   └── ...
│   └── labels/              # YOLO format labels (.txt)
│       ├── route_001.txt
│       ├── route_002.txt
│       └── ...
├── val/
│   ├── images/              # Validation images
│   └── labels/              # Validation labels
└── test/                    # Optional
    ├── images/
    └── labels/
```

#### data.yaml Configuration

```yaml
# data.yaml
train: train      # Path to training images (relative to this file)
val: val          # Path to validation images
test: test        # Optional test set

nc: 8             # Number of classes

names:            # Class names (must match nc count)
  0: crimp
  1: jug
  2: sloper
  3: pinch
  4: pocket
  5: foot-hold
  6: start-hold
  7: top-out-hold
```

#### Label Format (YOLO Format)

Each image has a corresponding `.txt` file with the same name:

```
# route_001.txt
# Format: <class_id> <x_center> <y_center> <width> <height>
# All values normalized to [0, 1] relative to image dimensions

0 0.5123 0.3456 0.0821 0.1234    # crimp
1 0.7234 0.6543 0.1456 0.2134    # jug
5 0.2345 0.8765 0.0987 0.1543    # foot-hold
```

**Normalization Example**:
```python
# Image size: 1920 x 1080
# Hold bounding box (pixels): x1=500, y1=300, x2=600, y2=400

x_center = (500 + 600) / 2 / 1920 = 0.2865
y_center = (300 + 400) / 2 / 1080 = 0.3241
width = (600 - 500) / 1920 = 0.0521
height = (400 - 300) / 1080 = 0.0926

# Label line:
# 0 0.2865 0.3241 0.0521 0.0926
```

### 2. Install Dependencies

```bash
# Install required packages
pip install ultralytics torch torchvision
pip install pyyaml flask sqlalchemy

# Or use requirements.txt
pip install -r requirements.txt
```

### 3. Run Training

#### Basic Training Command

```bash
python src/train_model.py --model-name v1.0
```

#### Advanced Training with Custom Parameters

```bash
python src/train_model.py \
    --model-name v1.1 \
    --data-yaml data/your_dataset/data.yaml \
    --base-weights yolov8n.pt \
    --epochs 100 \
    --batch-size 16 \
    --learning-rate 0.01 \
    --img-size 640 \
    --activate
```

**Parameters Explained**:
- `--model-name`: Version identifier for this model (e.g., "v1.0", "v2.0-crimp-focused")
- `--data-yaml`: Path to dataset configuration file
- `--base-weights`: Starting weights (default: yolov8n.pt)
- `--epochs`: Training iterations (default: 100)
- `--batch-size`: Batch size (-1 for auto, default: 16)
- `--learning-rate`: Initial learning rate (default: 0.01)
- `--img-size`: Input image size (default: 640)
- `--activate`: Set this model as active immediately

### 4. Training Process

The script will:

1. **Validate dataset** - Check data.yaml and directory structure
2. **Validate base weights** - Ensure model file exists
3. **Setup directories** - Create output folders
4. **Train model** - Run YOLOv8 training
5. **Save model** - Copy best weights and create database entry

**Output Structure**:
```
runs/detect/v1.0/
├── weights/
│   ├── best.pt              # Best performing weights
│   └── last.pt              # Final epoch weights
├── results.csv              # Training metrics per epoch
├── confusion_matrix.png     # Class confusion matrix
├── PR_curve.png            # Precision-recall curve
└── ...                      # Other plots and logs

models/hold_detection/
├── v1.0.pt                  # Copied best weights
└── v1.0_metadata.yaml       # Training configuration & metrics
```

### 5. Monitor Training

**Console Output**:
```
Epoch   GPU_mem   box_loss   cls_loss   dfl_loss   Instances   Size
1/100   2.3GB     1.234      0.567      0.891      156         640
2/100   2.3GB     1.123      0.543      0.876      156         640
...
```

**Key Metrics**:
- `box_loss`: Bounding box regression loss (lower is better)
- `cls_loss`: Classification loss (lower is better)
- `mAP@0.5`: Mean Average Precision at IoU threshold 0.5
- `mAP@0.5:0.95`: mAP averaged over IoU thresholds 0.5-0.95

**Good Training Indicators**:
- Losses decrease steadily
- mAP increases steadily
- No overfitting (train and val metrics stay close)

### 6. Activate Trained Model

**Option 1: During training** (use `--activate` flag):
```bash
python src/train_model.py --model-name v1.0 --activate
```

**Option 2: After training** (use model management script):
```bash
python src/manage_models.py activate v1.0
```

**Option 3: Manual database update**:
```sql
UPDATE model_versions 
SET is_active = 0 
WHERE model_type = 'hold_detection';

UPDATE model_versions 
SET is_active = 1 
WHERE version = 'v1.0' AND model_type = 'hold_detection';
```

---

## Adding Slant Angle Detection

### Problem: Current Model Doesn't Detect Orientation

**Current Detection Output**:
```python
{
    'bbox': [x1, y1, x2, y2],
    'class_id': 0,  # crimp
    'confidence': 0.92
}
# Missing: orientation angle
```

**Desired Output** (for slant detection):
```python
{
    'bbox': [x1, y1, x2, y2],
    'class_id': 0,  # crimp
    'confidence': 0.92,
    'orientation': 45.0,  # NEW: degrees from horizontal
    'slant_category': 'downward'  # NEW: downward/horizontal/upward
}
```

### Option 1: Oriented Bounding Box (OBB) Detection

**Approach**: Use YOLOv8-OBB instead of standard YOLOv8

**What Changes**:
1. Bounding box format includes rotation
2. Labels include angle information
3. Model architecture supports oriented detection

**Label Format Changes**:
```
# Standard YOLO (current):
<class_id> <x_center> <y_center> <width> <height>

# YOLO-OBB (for orientation):
<class_id> <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4>
# Or: <class_id> <x_center> <y_center> <width> <height> <angle>
```

**Example OBB Label**:
```
# Crimp tilted 30 degrees
0 0.5 0.3 0.08 0.12 30.0
```

**Implementation Steps**:

1. **Re-annotate Dataset with Orientation**:
   - Use tools like Roboflow or LabelImg with rotation support
   - Mark bounding boxes with rotation angle
   - Export in YOLO-OBB format

2. **Update data.yaml**:
   ```yaml
   # No changes needed - same format
   train: train
   val: val
   nc: 8
   names: [crimp, jug, sloper, pinch, pocket, foot-hold, start-hold, top-out-hold]
   ```

3. **Modify Training Script**:
   ```python
   # In src/train_model.py, line 297:
   from ultralytics import YOLO
   
   # Load OBB model instead of detection model
   model = YOLO('yolov8n-obb.pt')  # Use OBB variant
   ```

4. **Update Detection Code** (in src/main.py):
   ```python
   # Current detection (line ~520):
   results = model(image_np)
   
   # Extract detections
   for det in results[0].boxes.data:
       x1, y1, x2, y2, conf, cls = det
   
   # NEW: OBB detection
   for obb in results[0].obb.data:
       x1, y1, x2, y2, x3, y3, x4, y4, conf, cls = obb
       # Or: x_center, y_center, width, height, angle, conf, cls
       
       # Calculate orientation
       angle = calculate_obb_angle(x1, y1, x2, y2, x3, y3, x4, y4)
   ```

**Pros**:
- Native YOLO support
- Single model for both detection and orientation
- Good accuracy

**Cons**:
- Requires complete dataset re-annotation (100+ images)
- 2-4 weeks of annotation work
- More complex training setup

---

### Option 2: Two-Stage Approach

**Approach**: Detect holds first, then classify orientation

**Step 1**: Use current model for hold detection  
**Step 2**: Train separate orientation classifier

**Implementation**:

1. **Current Model**: Detects holds and types (no changes)

2. **New Orientation Classifier**:
   ```python
   # Train simple CNN for orientation
   # Input: Cropped hold image (from bbox)
   # Output: Orientation class (0=down, 1=horizontal, 2=up)
   
   class OrientationClassifier(nn.Module):
       def __init__(self):
           super().__init__()
           self.conv1 = nn.Conv2d(3, 32, 3)
           self.conv2 = nn.Conv2d(32, 64, 3)
           self.fc1 = nn.Linear(64*6*6, 128)
           self.fc2 = nn.Linear(128, 3)  # 3 orientation classes
   ```

3. **Pipeline**:
   ```python
   # Detect holds
   hold_detections = yolo_model(image)
   
   # For each hold, classify orientation
   for detection in hold_detections:
       hold_crop = crop_image(image, detection.bbox)
       orientation = orientation_classifier(hold_crop)
       detection.orientation = orientation
   ```

**Annotation Requirements**:
- Re-annotate with orientation labels (simpler than OBB)
- Just add orientation class to each hold

**Example Annotation CSV**:
```csv
image_id,bbox_x1,bbox_y1,bbox_x2,bbox_y2,hold_type,orientation
route_001,500,300,600,400,crimp,downward
route_001,700,500,850,650,jug,horizontal
```

**Pros**:
- Simpler than OBB
- Can train orientation model separately
- Less annotation work

**Cons**:
- Two-stage inference (slower)
- Two models to maintain
- Potential error compounding

---

### Option 3: Manual Annotation UI

**Approach**: Don't detect automatically, let users annotate

**Implementation**:

1. **Add UI in upload form**:
   ```html
   <div class="hold-annotation">
       <h4>Detected Holds - Add Orientation</h4>
       <div id="hold-list">
           <!-- For each detected hold -->
           <div class="hold-item">
               <img src="hold_crop_001.jpg">
               <span>Crimp #1</span>
               <select name="orientation_1">
                   <option value="unknown">Unknown</option>
                   <option value="downward">Downward ⬇️</option>
                   <option value="horizontal">Horizontal ↔️</option>
                   <option value="upward">Upward ⬆️</option>
               </select>
           </div>
       </div>
   </div>
   ```

2. **Store in database**:
   ```python
   # Add to DetectedHold model
   class DetectedHold(Base):
       # ... existing fields ...
       orientation = Column(String(20), nullable=True)
       # Values: 'downward', 'horizontal', 'upward', 'unknown'
   ```

**Pros**:
- No model retraining needed
- Accurate (human-labeled)
- Can implement immediately

**Cons**:
- Manual work per route
- User friction
- Inconsistent labeling

---

## Recommendation for Your Project

### For Phase 1a MVP: **Option 3 (Manual Annotation)**

**Why**:
- ✅ Can implement in 1-2 days
- ✅ No model retraining required
- ✅ Provides data for future automatic detection
- ✅ Validates if slant detection is actually critical

**Implementation**:
```python
# 1. Add orientation field to DetectedHold model
# In src/models.py
orientation = Column(String(20), nullable=True, default='unknown')

# 2. Add UI dropdowns for each detected hold
# In src/templates/index.html (results section)

# 3. Save orientation when user submits
# In src/main.py analysis endpoint
```

### For Phase 1b: **Option 1 (YOLO-OBB)** - IF slant proves critical

**Prerequisites**:
- Phase 1a feedback shows slant is important
- Budget for 2-4 weeks of dataset annotation
- Clear accuracy improvement expected

**Steps**:
1. Annotate 100+ routes with oriented bounding boxes
2. Train YOLOv8-OBB model
3. Update detection pipeline
4. Test accuracy improvement

### For Phase 2: **Option 2 (Two-Stage)** - IF OBB too expensive

**When**:
- OBB annotation too costly
- Need faster iteration
- Orientation less critical than hold detection

---

## Training Best Practices

### 1. Dataset Quality

**Minimum Dataset Size**:
- Training: ≥100 images (more is better)
- Validation: ≥20 images
- Per-class balance: ≥10 examples per hold type

**Image Quality**:
- Resolution: 1280x720 or higher
- Good lighting
- Clear hold visibility
- Varied angles and routes

**Annotation Quality**:
- Tight bounding boxes
- Consistent labeling
- Review for errors

### 2. Training Parameters

**For Small Datasets (100-300 images)**:
```bash
python src/train_model.py \
    --model-name v1.0 \
    --epochs 150 \
    --batch-size 8 \
    --learning-rate 0.005
```

**For Medium Datasets (300-1000 images)**:
```bash
python src/train_model.py \
    --model-name v1.0 \
    --epochs 100 \
    --batch-size 16 \
    --learning-rate 0.01
```

**For Large Datasets (1000+ images)**:
```bash
python src/train_model.py \
    --model-name v1.0 \
    --epochs 80 \
    --batch-size 32 \
    --learning-rate 0.01
```

### 3. Monitoring Training

**Check for Overfitting**:
```
# Good: train and val losses close
Epoch 50: train_loss=0.45, val_loss=0.48  ✅

# Bad: val loss much higher than train
Epoch 50: train_loss=0.25, val_loss=0.65  ❌ Overfitting!
```

**Solutions for Overfitting**:
- Add more training data
- Use data augmentation
- Reduce model complexity
- Add regularization

### 4. Model Evaluation

**After Training, Evaluate**:
```python
from ultralytics import YOLO

# Load trained model
model = YOLO('models/hold_detection/v1.0.pt')

# Run validation
results = model.val(data='data/your_dataset/data.yaml')

# Check metrics
print(f"mAP@0.5: {results.box.map50}")
print(f"mAP@0.5:0.95: {results.box.map}")
print(f"Precision: {results.box.mp}")
print(f"Recall: {results.box.mr}")
```

**Target Metrics**:
- mAP@0.5: ≥0.80 (good), ≥0.90 (excellent)
- mAP@0.5:0.95: ≥0.60 (good), ≥0.75 (excellent)
- Precision: ≥0.85
- Recall: ≥0.80

---

## Common Issues & Solutions

### Issue 1: "No images found in train/images"

**Solution**:
```bash
# Check directory structure
ls -R data/your_dataset/

# Ensure structure matches:
# train/images/*.jpg
# train/labels/*.txt
```

### Issue 2: "Class mismatch - nc=8 but found class 9"

**Solution**:
- Check label files for class IDs > 7
- Class IDs must be 0-7 for 8 classes
- Fix annotation errors

### Issue 3: Low mAP (<0.5)

**Possible Causes**:
- Too few training images
- Poor annotation quality
- Imbalanced classes
- Insufficient training epochs

**Solutions**:
- Add more training data
- Review and fix annotations
- Balance class distribution
- Train longer (150-200 epochs)

### Issue 4: Out of Memory (CUDA OOM)

**Solution**:
```bash
# Reduce batch size
python src/train_model.py --batch-size 8

# Or use CPU (slower)
python src/train_model.py --device cpu
```

---

## Summary

### Current Setup
- ✅ Detects 8 hold types
- ✅ Provides bounding boxes
- ✅ Classifies hold type
- ❌ Does NOT detect orientation/slant

### To Add Slant Detection

**Phase 1a MVP**: Manual annotation (1-2 days)  
**Phase 1b**: YOLO-OBB training (2-4 weeks)  
**Alternative**: Two-stage classifier (1-2 weeks)

### Next Steps

1. **Try current training** with sample dataset
2. **Collect Phase 1a feedback** on whether slant is critical
3. **Decide** which slant detection approach based on feedback
4. **Annotate dataset** if automatic detection needed
5. **Retrain model** with orientation support

---

## Quick Start Commands

```bash
# 1. Prepare your dataset
mkdir -p data/my_holds/{train,val}/{images,labels}
# Add your images and labels

# 2. Create data.yaml
cat > data/my_holds/data.yaml << EOF
train: train
val: val
nc: 8
names: [crimp, jug, sloper, pinch, pocket, foot-hold, start-hold, top-out-hold]
EOF

# 3. Train model
python src/train_model.py \
    --model-name my_v1 \
    --data-yaml data/my_holds/data.yaml \
    --epochs 100 \
    --activate

# 4. Test model
python -c "
from ultralytics import YOLO
model = YOLO('models/hold_detection/my_v1.pt')
results = model.predict('test_image.jpg', save=True)
"
```

---

**Questions?** See the [Training Script](../src/train_model.py) for implementation details.
