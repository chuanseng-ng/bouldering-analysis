from datetime import datetime, timezone
import uuid
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Analysis(db.Model):
    """Stores analysis results for uploaded images"""

    __tablename__ = "analyses"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    image_filename = db.Column(db.String(255), nullable=False)
    image_path = db.Column(db.String(500), nullable=False)
    predicted_grade = db.Column(db.String(10), nullable=False)
    confidence_score = db.Column(db.Float, nullable=True)
    holds_detected = db.Column(db.JSON, nullable=True)  # Store hold detection results
    features_extracted = db.Column(db.JSON, nullable=True)  # Store extracted features
    created_at = db.Column(db.DateTime, default=datetime.now(timezone.utc))
    updated_at = db.Column(
        db.DateTime,
        default=datetime.now(timezone.utc),
        onupdate=datetime.now(timezone.utc),
    )

    # Relationship to feedback
    feedback = db.relationship("Feedback", backref="analysis", uselist=False, lazy=True)

    # Relationship to detected holds
    detected_holds = db.relationship("DetectedHold", backref="analysis", lazy=True)

    # Indexes for common queries
    __table_args__ = (
        db.Index("idx_analysis_predicted_grade", "predicted_grade"),
        db.Index("idx_analysis_created_at", "created_at"),
    )


class Feedback(db.Model):
    """Stores user feedback on analysis results"""

    __tablename__ = "feedback"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    analysis_id = db.Column(db.String(36), db.ForeignKey("analyses.id"), nullable=False)
    user_grade = db.Column(db.String(10), nullable=True)  # User's proposed grade
    is_accurate = db.Column(
        db.Boolean, nullable=False, default=False
    )  # Whether user agreed with prediction
    comments = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now(timezone.utc))

    # Index for faster queries
    __table_args__ = (
        db.Index("idx_feedback_analysis_id", "analysis_id"),
        db.Index("idx_feedback_created_at", "created_at"),
    )


class HoldType(db.Model):
    """Reference table for hold types"""

    __tablename__ = "hold_types"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    description = db.Column(db.Text, nullable=True)

    # Relationship to detected holds
    detected_holds = db.relationship("DetectedHold", backref="hold_type", lazy=True)


class DetectedHold(db.Model):
    """Stores individual hold detections"""

    __tablename__ = "detected_holds"

    id = db.Column(db.Integer, primary_key=True)
    analysis_id = db.Column(db.String(36), db.ForeignKey("analyses.id"), nullable=False)
    hold_type_id = db.Column(db.Integer, db.ForeignKey("hold_types.id"), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    bbox_x1 = db.Column(db.Float, nullable=False)
    bbox_y1 = db.Column(db.Float, nullable=False)
    bbox_x2 = db.Column(db.Float, nullable=False)
    bbox_y2 = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now(timezone.utc))

    # Index for faster queries
    __table_args__ = (
        db.Index("idx_detected_hold_analysis_id", "analysis_id"),
        db.Index("idx_detected_hold_hold_type_id", "hold_type_id"),
    )


class ModelVersion(db.Model):
    """Tracks different versions of trained models"""

    __tablename__ = "model_versions"

    id = db.Column(db.Integer, primary_key=True)
    model_type = db.Column(
        db.String(50), nullable=False
    )  # 'hold_detection' or 'route_grading'
    version = db.Column(db.String(20), nullable=False)
    model_path = db.Column(db.String(500), nullable=False)
    accuracy = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now(timezone.utc))
    is_active = db.Column(db.Boolean, default=True)

    # Index for faster queries
    __table_args__ = (
        db.Index("idx_model_version_type", "model_type"),
        db.Index("idx_model_version_active", "is_active"),
        db.UniqueConstraint("model_type", "version", name="uq_model_type_version"),
    )


class UserSession(db.Model):
    """Tracks user sessions for analytics"""

    __tablename__ = "user_sessions"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = db.Column(db.String(36), nullable=False, unique=True)
    ip_address = db.Column(db.String(45), nullable=True)
    user_agent = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now(timezone.utc))
    last_activity = db.Column(db.DateTime, default=datetime.now(timezone.utc))

    # Index for faster queries
    __table_args__ = (db.Index("idx_user_session_created_at", "created_at"),)
