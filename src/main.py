import os
import uuid
from datetime import datetime, timezone
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__, template_folder="templates")

# Configuration
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY")
if not app.config["SECRET_KEY"]:
    if os.environ.get("FLASK_ENV") == "production":
        raise ValueError("SECRET_KEY must be set in production")
    app.config["SECRET_KEY"] = "dev-secret-key-change-in-production"
app.config["SQLALCHEMY_DATABASE_URI"] = (
    os.environ.get("DATABASE_URL") or "sqlite:///bouldering_analysis.db"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["UPLOAD_FOLDER"] = "data/uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg"}

# Import models and db after app creation
# pylint: disable=import-outside-toplevel, wrong-import-position
from src.models import db  # noqa: E402

# Initialize extensions
db.init_app(app)

# Import models after db is initialized
# pylint: disable=import-outside-toplevel, wrong-import-position
from src.models import (  # noqa: E402
    Analysis,
    Feedback,
    HoldType,
    DetectedHold,
)

# Load models
try:
    hold_detection_model = YOLO("yolov8n.pt")
    print("YOLOv8 model loaded successfully")
except Exception as e:
    print(f"Error loading YOLOv8 model: {e}")
    hold_detection_model = None

# Hold type mapping (this should be populated from the database)
HOLD_TYPES = {
    0: "crimp",
    1: "jug",
    2: "sloper",
    3: "pinch",
    4: "pocket",
    5: "foot-hold",
    6: "start-hold",
    7: "top-out-hold",
}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]
    )


def create_tables():
    """Create database tables"""
    with app.app_context():
        db.create_all()

        # Initialize hold types if they don't exist
        if HoldType.query.count() == 0:
            hold_type_data = [
                (0, "crimp", "Small, narrow hold requiring crimping fingers"),
                (1, "jug", "Large, easy-to-hold jug"),
                (2, "sloper", "Round, sloping hold that requires open-handed grip"),
                (3, "pinch", "Hold that requires pinching between thumb and fingers"),
                (4, "pocket", "Small hole that fingers fit into"),
                (5, "foot-hold", "Hold specifically for feet"),
                (6, "start-hold", "Starting hold for the route"),
                (7, "top-out-hold", "Hold used to complete the route"),
            ]

            for hold_id, name, description in hold_type_data:
                hold_type = HoldType(id=hold_id, name=name, description=description)
                db.session.add(hold_type)

            db.session.commit()
            print("Hold types initialized")


@app.route("/", methods=["GET", "POST"])
def index():
    """Main page with image upload form"""
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", error="No file part")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", error="No selected file")

        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
            os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
            file.save(filepath)

            # Process the image
            try:
                result = analyze_image(filepath, unique_filename)
                return render_template(
                    "index.html", result=result, image_path=unique_filename
                )
            except Exception as e:
                return render_template(
                    "index.html", error=f"Error processing image: {str(e)}"
                )
        else:
            return render_template(
                "index.html",
                error="Invalid file type. Please upload PNG, JPG, or JPEG images.",
            )

    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze_route():
    """API endpoint for analyzing a bouldering route"""
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if file and allowed_file(file.filename):
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        file.save(filepath)

        # Process the image
        try:
            result = analyze_image(filepath, unique_filename)
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": f"Error processing image: {str(e)}"}), 500
    else:
        return (
            jsonify(
                {"error": "Invalid file type. Please upload PNG, JPG, or JPEG images."}
            ),
            400,
        )


@app.route("/feedback", methods=["POST"])
def submit_feedback():
    """API endpoint for submitting user feedback"""
    data = request.get_json()

    if not data or "analysis_id" not in data:
        return jsonify({"error": "Missing analysis_id"}), 400

    try:
        analysis = db.session.get(Analysis, data["analysis_id"])
        if not analysis:
            return jsonify({"error": "Analysis not found"}), 404

        # Create feedback record
        feedback = Feedback(
            analysis_id=data["analysis_id"],
            user_grade=data.get("user_grade"),
            is_accurate=data.get("is_accurate", False),
            comments=data.get("comments"),
        )

        db.session.add(feedback)
        db.session.commit()

        return jsonify(
            {"message": "Feedback submitted successfully", "feedback_id": feedback.id}
        )

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Error submitting feedback: {str(e)}"}), 500


@app.route("/stats", methods=["GET"])
def get_stats():
    """API endpoint for getting usage statistics"""
    try:
        total_analyses = Analysis.query.count()
        total_feedback = Feedback.query.count()
        accurate_predictions = Feedback.query.filter_by(is_accurate=True).count()

        # Get grade distribution
        grade_counts = (
            db.session.query(Analysis.predicted_grade, db.func.count(Analysis.id))
            .group_by(Analysis.predicted_grade)
            .all()
        )

        stats = {
            "total_analyses": total_analyses,
            "total_feedback": total_feedback,
            "accurate_predictions": accurate_predictions,
            "accuracy_rate": (
                (accurate_predictions / total_feedback * 100)
                if total_feedback > 0
                else 0
            ),
            "grade_distribution": {grade: count for grade, count in grade_counts},
        }

        return jsonify(stats)

    except Exception as e:
        return jsonify({"error": f"Error getting stats: {str(e)}"}), 500


def check_db_connection():
    """Check database connectivity"""
    try:
        from sqlalchemy import text

        db.session.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    model_ok = hold_detection_model is not None
    db_ok = check_db_connection()
    overall_ok = model_ok and db_ok

    status = {
        "status": "healthy" if overall_ok else "unhealthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_loaded": model_ok,
        "database_connected": db_ok,
    }
    return jsonify(status), 200 if overall_ok else 503


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


def analyze_image(image_path, image_filename):
    """Analyze a bouldering route image"""
    if not hold_detection_model:
        raise RuntimeError("Hold detection model not loaded")

    # Load and preprocess image
    img = Image.open(image_path)
    img = img.convert("RGB")

    # Run detection
    results = hold_detection_model(img)

    # Process results
    holds = []  # Keep for API response compatibility
    features = {"total_holds": 0, "hold_types": {}, "average_confidence": 0}

    total_confidence = 0.0
    detected_holds_data = []  # Temporary storage for hold data

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())

                # Map class ID to hold type
                hold_type = HOLD_TYPES.get(class_id, "unknown")

                # Store data for DetectedHold creation
                detected_holds_data.append(
                    {
                        "hold_type": hold_type,
                        "confidence": float(confidence),
                        "bbox_x1": float(x1),
                        "bbox_y1": float(y1),
                        "bbox_x2": float(x2),
                        "bbox_y2": float(y2),
                    }
                )

                # Keep for API response
                hold_data = {
                    "type": hold_type,
                    "confidence": float(confidence),
                    "bbox": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                    },
                }
                holds.append(hold_data)
                total_confidence += float(confidence)

                # Update features
                features["total_holds"] += 1
                features["hold_types"][hold_type] = (
                    features["hold_types"].get(hold_type, 0) + 1
                )

    # Calculate average confidence
    if holds:
        features["average_confidence"] = float(total_confidence) / len(holds)
    else:
        features["average_confidence"] = 0

    # Predict grade
    predicted_grade = predict_grade(features)

    # Create Analysis record (without holds_detected JSON)
    analysis = Analysis(
        image_filename=image_filename,
        image_path=image_path,
        predicted_grade=predicted_grade,
        confidence_score=features["average_confidence"],
        features_extracted=features,
    )
    db.session.add(analysis)
    db.session.flush()  # Flush to get the analysis.id before creating DetectedHold records

    # Create DetectedHold records
    for hold_data in detected_holds_data:
        # Look up HoldType by name
        hold_type = HoldType.query.filter_by(name=hold_data["hold_type"]).first()

        if hold_type:  # Only create if valid hold type exists
            detected_hold = DetectedHold(
                analysis_id=analysis.id,
                hold_type_id=hold_type.id,
                confidence=hold_data["confidence"],
                bbox_x1=hold_data["bbox_x1"],
                bbox_y1=hold_data["bbox_y1"],
                bbox_x2=hold_data["bbox_x2"],
                bbox_y2=hold_data["bbox_y2"],
            )
            db.session.add(detected_hold)

    db.session.commit()

    # Query back the detected holds for response
    detected_holds_query = DetectedHold.query.filter_by(analysis_id=analysis.id).all()
    holds_from_db = [
        {
            "type": dh.hold_type.name,
            "confidence": dh.confidence,
            "bbox": {
                "x1": dh.bbox_x1,
                "y1": dh.bbox_y1,
                "x2": dh.bbox_x2,
                "y2": dh.bbox_y2,
            },
        }
        for dh in detected_holds_query
    ]

    return {
        "analysis_id": analysis.id,
        "predicted_grade": predicted_grade,
        "confidence": features["average_confidence"],
        "holds": holds_from_db,
        "features": features,
    }


def predict_grade(features):
    """Predict V-grade based on extracted features (simplified)"""
    # This is a simplified grading algorithm
    # In production, this would use a trained machine learning model

    hold_count = features["total_holds"]
    hold_types = features["hold_types"]
    # avg_confidence = features["average_confidence"] # TODO: Incorporate confidence into grading

    # Base grade on hold count
    if hold_count <= 3:
        base_grade = "V0"
    elif hold_count <= 5:
        base_grade = "V1"
    elif hold_count <= 7:
        base_grade = "V2"
    elif hold_count <= 9:
        base_grade = "V3"
    elif hold_count <= 12:
        base_grade = "V4"
    else:
        base_grade = "V5"

    # Adjust based on hold types
    difficulty_multiplier = 1.0

    # Small holds increase difficulty
    if hold_types.get("crimp", 0) > 0:
        difficulty_multiplier += 0.2 * hold_types["crimp"]
    if hold_types.get("pocket", 0) > 0:
        difficulty_multiplier += 0.3 * hold_types["pocket"]
    if hold_types.get("sloper", 0) > 0:
        difficulty_multiplier += 0.1 * hold_types["sloper"]

    # Adjust grade based on difficulty
    grade_value = int(base_grade[1:])
    grade_value = min(grade_value + int(difficulty_multiplier), 10)  # Cap at V10

    return f"V{grade_value}"


if __name__ == "__main__":
    # Create tables before running the app
    create_tables()

    # Run the application
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", 5000))
    app.run(debug=debug_mode, host=host, port=port)
