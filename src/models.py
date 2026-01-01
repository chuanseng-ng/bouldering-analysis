"""
This module defines the database models for the bouldering analysis application.

It includes models for storing analysis results, user feedback, hold types,
detected holds, model versions, and user sessions.
"""

from datetime import datetime, timezone
import uuid
from typing import Any
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Base(db.Model):  # type: ignore[name-defined]
    """Base class for all SQLAlchemy models"""

    __abstract__ = True

    def __repr__(self) -> str:
        """Return a string representation of the model."""
        return f"<{self.__class__.__name__}>"

    def to_dict(self) -> dict[str, Any]:
        """Convert the model to a dictionary."""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


def utcnow():
    """Return current UTC datetime with timezone awareness."""
    return datetime.now(timezone.utc)


class Analysis(Base):
    """Stores analysis results for uploaded images"""

    __tablename__ = "analyses"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    image_filename = db.Column(db.String(255), nullable=False)
    image_path = db.Column(db.String(500), nullable=False)
    predicted_grade = db.Column(db.String(10), nullable=False)
    confidence_score = db.Column(db.Float, nullable=True)
    holds_detected = db.Column(db.JSON, nullable=True)  # Store hold detection results
    features_extracted = db.Column(db.JSON, nullable=True)  # Store extracted features
    created_at = db.Column(db.DateTime, default=utcnow)
    updated_at = db.Column(
        db.DateTime,
        default=utcnow,
        onupdate=utcnow,
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

    # No custom __init__ needed - SQLAlchemy column default handles UUID generation

    def __repr__(self):
        """Return a string representation of the Analysis object."""
        return f"<Analysis {self.id}: {self.image_filename} - Grade {self.predicted_grade}>"

    def to_dict(self) -> dict[str, Any]:
        """Convert the Analysis object to a dictionary."""
        return {
            "id": self.id,
            "image_filename": self.image_filename,
            "image_path": self.image_path,
            "predicted_grade": self.predicted_grade,
            "confidence_score": self.confidence_score,
            "holds_detected": self.holds_detected,
            "features_extracted": self.features_extracted,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Feedback(Base):
    """Stores user feedback on analysis results"""

    __tablename__ = "feedback"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    analysis_id = db.Column(db.String(36), db.ForeignKey("analyses.id"), nullable=False)
    user_grade = db.Column(db.String(10), nullable=True)  # User's proposed grade
    is_accurate = db.Column(
        db.Boolean, nullable=False, default=False
    )  # Whether user agreed with prediction
    comments = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=utcnow)

    # Index for faster queries
    __table_args__ = (
        db.Index("idx_feedback_analysis_id", "analysis_id"),
        db.Index("idx_feedback_created_at", "created_at"),
    )

    # No custom __init__ needed - SQLAlchemy column default handles UUID generation

    def __repr__(self):
        """Return a string representation of the Feedback object."""
        return f"<Feedback {self.id}: Analysis {self.analysis_id} - Accurate {self.is_accurate}>"

    def to_dict(self) -> dict[str, Any]:
        """Convert the Feedback object to a dictionary."""
        return {
            "id": self.id,
            "analysis_id": self.analysis_id,
            "user_grade": self.user_grade,
            "is_accurate": self.is_accurate,
            "comments": self.comments,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class HoldType(Base):
    """Reference table for hold types"""

    __tablename__ = "hold_types"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    description = db.Column(db.Text, nullable=True)

    # Relationship to detected holds
    detected_holds = db.relationship("DetectedHold", backref="hold_type", lazy=True)

    def __repr__(self):
        """Return a string representation of the HoldType object."""
        return f"<HoldType {self.id}: {self.name}>"

    def to_dict(self) -> dict[str, Any]:
        """Convert the HoldType object to a dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
        }


class DetectedHold(Base):
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
    created_at = db.Column(db.DateTime, default=utcnow)

    # Index for faster queries
    __table_args__ = (
        db.Index("idx_detected_hold_analysis_id", "analysis_id"),
        db.Index("idx_detected_hold_hold_type_id", "hold_type_id"),
    )

    def __repr__(self):
        """Return a string representation of the DetectedHold object."""
        return f"<DetectedHold {self.id}: Analysis {self.analysis_id} - Type {self.hold_type_id}>"

    def to_dict(self) -> dict[str, Any]:
        """Convert the DetectedHold object to a dictionary."""
        return {
            "id": self.id,
            "analysis_id": self.analysis_id,
            "hold_type_id": self.hold_type_id,
            "confidence": self.confidence,
            "bbox": {
                "x1": self.bbox_x1,
                "y1": self.bbox_y1,
                "x2": self.bbox_x2,
                "y2": self.bbox_y2,
            },
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ModelVersion(Base):
    """Tracks different versions of trained models"""

    __tablename__ = "model_versions"

    id = db.Column(db.Integer, primary_key=True)
    model_type = db.Column(
        db.String(50), nullable=False
    )  # 'hold_detection' or 'route_grading'
    version = db.Column(db.String(20), nullable=False)
    model_path = db.Column(db.String(500), nullable=False)
    accuracy = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=utcnow)
    is_active = db.Column(db.Boolean, default=True)

    # Index for faster queries
    __table_args__ = (
        db.Index("idx_model_version_type", "model_type"),
        db.Index("idx_model_version_active", "is_active"),
        db.UniqueConstraint("model_type", "version", name="uq_model_type_version"),
    )

    # No custom __init__ needed - SQLAlchemy handles primary key generation

    def __repr__(self):
        """Return a string representation of the ModelVersion object."""
        return f"<ModelVersion {self.id}: {self.model_type} v{self.version} - Active {self.is_active}>"

    def to_dict(self) -> dict[str, Any]:
        """Convert the ModelVersion object to a dictionary."""
        return {
            "id": self.id,
            "model_type": self.model_type,
            "version": self.version,
            "model_path": self.model_path,
            "accuracy": self.accuracy,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "is_active": self.is_active,
        }


class UserSession(Base):
    """Tracks user sessions for analytics"""

    __tablename__ = "user_sessions"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id = db.Column(db.String(36), nullable=False, unique=True)
    ip_address = db.Column(db.String(45), nullable=True)
    user_agent = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=utcnow)
    last_activity = db.Column(db.DateTime, default=utcnow)

    # Index for faster queries
    __table_args__ = (db.Index("idx_user_session_created_at", "created_at"),)

    # No custom __init__ needed - SQLAlchemy column default handles UUID generation

    def __repr__(self):
        """Return a string representation of the UserSession object."""
        return f"<UserSession {self.id}: {self.session_id}>"

    def to_dict(self) -> dict[str, Any]:
        """Convert the UserSession object to a dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_activity": (
                self.last_activity.isoformat() if self.last_activity else None
            ),
        }
