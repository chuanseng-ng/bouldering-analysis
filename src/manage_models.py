"""
Model management utility for activating/deactivating model versions.

This module provides functions to manage different versions of trained models
in the bouldering analysis application. It allows activation and deactivation
of specific model versions, listing all models, and retrieving the currently
active model.

Usage Examples:
    List all models:
        python src/manage_models.py list

    List hold detection models only:
        python src/manage_models.py list --model-type hold_detection

    Activate a specific model version:
        python src/manage_models.py activate --model-type hold_detection --version v1.0

    Deactivate a specific model version:
        python src/manage_models.py deactivate --model-type hold_detection --version v1.0

Functions can also be imported and used programmatically:
    from src.manage_models import activate_model, list_models

    # Activate a model
    success, message = activate_model("hold_detection", "v1.0")

    # List all models
    models_info = list_models()
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

from src.config import resolve_path

# Add project root to Python path for imports
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _setup_flask_app():
    """
    Set up Flask application context for database operations.

    This is required because the database models are tied to Flask-SQLAlchemy.

    Returns:
        Flask app instance with configured database

    Raises:
        RuntimeError: If Flask app or database cannot be initialized
    """
    try:
        from flask import Flask  # pylint: disable=import-outside-toplevel
        from src.models import db  # pylint: disable=import-outside-toplevel

        app = Flask(__name__)

        # Configure database
        database_url = os.environ.get(
            "DATABASE_URL", "sqlite:///bouldering_analysis.db"
        )
        app.config["SQLALCHEMY_DATABASE_URI"] = database_url
        app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

        # Initialize database
        db.init_app(app)

        # Create tables if they don't exist
        with app.app_context():
            db.create_all()

        return app
    except ImportError as e:
        raise RuntimeError(f"Failed to import Flask dependencies: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Flask app: {e}") from e


def activate_model(model_type: str, version: str) -> Tuple[bool, str]:
    """
    Activate a specific model version.

    This function finds the specified model version in the database,
    deactivates all other models of the same type, and activates the
    specified model. It also validates that the model file exists at
    the specified path.

    Args:
        model_type: Type of model (e.g., 'hold_detection', 'route_grading')
        version: Version string of the model to activate

    Returns:
        Tuple[bool, str]: (success, message) where success is True if the
                         operation succeeded, and message contains details

    Examples:
        >>> success, msg = activate_model("hold_detection", "v1.0")
        >>> if success:
        ...     print(f"Model activated: {msg}")
        ... else:
        ...     print(f"Error: {msg}")
    """
    app = None
    try:
        # Set up Flask app and database
        app = _setup_flask_app()

        with app.app_context():
            from src.models import (  # pylint: disable=import-outside-toplevel
                db,
                ModelVersion,
            )

            # Find the specified model version
            target_model = (
                db.session.query(ModelVersion)
                .filter_by(model_type=model_type, version=version)
                .first()
            )

            if not target_model:
                error_msg = (
                    f"Model not found: model_type='{model_type}', version='{version}'"
                )
                logger.error(error_msg)
                return False, error_msg

            # Validate that the model file exists
            model_path = resolve_path(target_model.model_path)
            if not model_path.exists():
                error_msg = (
                    f"Model file not found at path: {model_path}\n"
                    f"Model: {model_type} v{version}"
                )
                logger.error(error_msg)
                return False, error_msg

            # Deactivate all other models of the same type
            other_models = (
                db.session.query(ModelVersion)
                .filter(
                    ModelVersion.model_type == model_type,
                    ModelVersion.id != target_model.id,
                )
                .all()
            )

            deactivated_count = 0
            for model in other_models:
                if model.is_active:
                    model.is_active = False
                    deactivated_count += 1
                    logger.info(
                        "Deactivated model: %s v%s (id=%d)",
                        model.model_type,
                        model.version,
                        model.id,
                    )

            # Activate the target model
            was_already_active = target_model.is_active
            target_model.is_active = True

            # Commit the changes
            db.session.commit()

            if was_already_active:
                success_msg = (
                    f"Model {model_type} v{version} was already active "
                    f"(deactivated {deactivated_count} other model(s))"
                )
            else:
                success_msg = (
                    f"Successfully activated model {model_type} v{version} "
                    f"(deactivated {deactivated_count} other model(s))"
                )

            logger.info(success_msg)
            return True, success_msg

    except Exception as e:  # pylint: disable=broad-exception-caught
        error_msg = f"Error activating model: {str(e)}"
        logger.exception(error_msg)

        # Attempt to rollback on error
        if app:
            try:
                with app.app_context():
                    from src.models import db  # pylint: disable=import-outside-toplevel

                    db.session.rollback()
            except (  # pylint: disable=broad-exception-caught
                Exception
            ) as rollback_error:
                logger.error("Failed to rollback transaction: %s", str(rollback_error))

        return False, error_msg


def deactivate_model(model_type: str, version: str) -> Tuple[bool, str]:
    """
    Deactivate a specific model version.

    This function finds and deactivates the specified model version.

    Args:
        model_type: Type of model (e.g., 'hold_detection', 'route_grading')
        version: Version string of the model to deactivate

    Returns:
        Tuple[bool, str]: (success, message) where success is True if the
                         operation succeeded, and message contains details

    Examples:
        >>> success, msg = deactivate_model("hold_detection", "v1.0")
        >>> if success:
        ...     print(f"Model deactivated: {msg}")
        ... else:
        ...     print(f"Error: {msg}")
    """
    app = None
    try:
        # Set up Flask app and database
        app = _setup_flask_app()

        with app.app_context():
            from src.models import (  # pylint: disable=import-outside-toplevel
                db,
                ModelVersion,
            )

            # Find the specified model version
            target_model = (
                db.session.query(ModelVersion)
                .filter_by(model_type=model_type, version=version)
                .first()
            )

            if not target_model:
                error_msg = (
                    f"Model not found: model_type='{model_type}', version='{version}'"
                )
                logger.error(error_msg)
                return False, error_msg

            # Deactivate the model
            was_active = target_model.is_active
            target_model.is_active = False

            # Commit the changes
            db.session.commit()

            if was_active:
                success_msg = f"Successfully deactivated model {model_type} v{version}"
            else:
                success_msg = f"Model {model_type} v{version} was already inactive"

            logger.info(success_msg)
            return True, success_msg

    except Exception as e:  # pylint: disable=broad-exception-caught
        error_msg = f"Error deactivating model: {str(e)}"
        logger.exception(error_msg)

        # Attempt to rollback on error
        if app:
            try:
                with app.app_context():
                    from src.models import db  # pylint: disable=import-outside-toplevel

                    db.session.rollback()
            except (  # pylint: disable=broad-exception-caught
                Exception
            ) as rollback_error:
                logger.error("Failed to rollback transaction: %s", str(rollback_error))

        return False, error_msg


def get_active_model(model_type: str) -> Optional[Any]:
    """
    Get the currently active model for a given model type.

    Args:
        model_type: Type of model (e.g., 'hold_detection', 'route_grading')

    Returns:
        ModelVersion object if an active model is found, None otherwise

    Examples:
        >>> active_model = get_active_model("hold_detection")
        >>> if active_model:
        ...     print(f"Active model: v{active_model.version}")
        ... else:
        ...     print("No active model found")
    """
    try:
        # Set up Flask app and database
        app = _setup_flask_app()

        with app.app_context():
            from src.models import (  # pylint: disable=import-outside-toplevel
                db,
                ModelVersion,
            )

            active_model = (
                db.session.query(ModelVersion)
                .filter_by(model_type=model_type, is_active=True)
                .first()
            )

            return active_model

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.exception("Error getting active model: %s", str(e))
        return None


def list_models(  # pylint: disable=too-many-locals
    model_type: Optional[str] = None,
) -> str:
    """
    List all model versions, optionally filtered by model_type.

    Displays version, model_type, accuracy, is_active, created_at, and model_path
    for each model. Highlights which model is currently active.

    Args:
        model_type: Optional filter for model type. If None, lists all models.

    Returns:
        Formatted string containing the model listing, or error message

    Examples:
        >>> # List all models
        >>> print(list_models())

        >>> # List only hold detection models
        >>> print(list_models("hold_detection"))
    """
    try:
        # Set up Flask app and database
        app = _setup_flask_app()

        with app.app_context():
            from src.models import (  # pylint: disable=import-outside-toplevel
                db,
                ModelVersion,
            )

            # Query models
            query = db.session.query(ModelVersion)
            if model_type:
                query = query.filter_by(model_type=model_type)

            models = query.order_by(
                ModelVersion.model_type, ModelVersion.created_at.desc()
            ).all()

            if not models:
                if model_type:
                    return f"No models found for model_type='{model_type}'"
                return "No models found in database"

            # Format the output
            lines = []
            lines.append("\n" + "=" * 100)
            lines.append("MODEL VERSIONS")
            lines.append("=" * 100)

            current_type = None
            for model in models:
                # Add separator between different model types
                if current_type != model.model_type:
                    if current_type is not None:
                        lines.append("-" * 100)
                    current_type = model.model_type

                # Format active indicator
                active_indicator = "  [ACTIVE]" if model.is_active else ""

                # Format accuracy
                accuracy_str = (
                    f"{model.accuracy:.4f}" if model.accuracy is not None else "N/A"
                )

                # Format created_at
                created_str = (
                    model.created_at.strftime("%Y-%m-%d %H:%M:%S")
                    if model.created_at
                    else "N/A"
                )

                # Check if model file exists
                model_path = resolve_path(model.model_path)
                file_exists = "[OK]" if model_path.exists() else "[FILE NOT FOUND]"

                lines.append(f"\nID:           {model.id}")
                lines.append(f"Type:         {model.model_type}{active_indicator}")
                lines.append(f"Version:      {model.version}")
                lines.append(f"Accuracy:     {accuracy_str}")
                lines.append(f"Created:      {created_str}")
                lines.append(f"Model Path:   {model.model_path} {file_exists}")

            lines.append("=" * 100)
            lines.append(f"\nTotal models: {len(models)}")
            active_count = sum(1 for m in models if m.is_active)
            lines.append(f"Active models: {active_count}\n")

            return "\n".join(lines)

    except Exception as e:  # pylint: disable=broad-exception-caught
        error_msg = f"Error listing models: {str(e)}"
        logger.exception(error_msg)
        return error_msg


def get_models_data(model_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get model version data as a list of dictionaries.

    This function is useful for programmatic access to model data.

    Args:
        model_type: Optional filter for model type. If None, returns all models.

    Returns:
        List of dictionaries containing model data, or empty list on error

    Examples:
        >>> models = get_models_data("hold_detection")
        >>> for model in models:
        ...     print(f"{model['version']}: {model['is_active']}")
    """
    try:
        # Set up Flask app and database
        app = _setup_flask_app()

        with app.app_context():
            from src.models import (  # pylint: disable=import-outside-toplevel
                db,
                ModelVersion,
            )

            # Query models
            query = db.session.query(ModelVersion)
            if model_type:
                query = query.filter_by(model_type=model_type)

            models = query.order_by(
                ModelVersion.model_type, ModelVersion.created_at.desc()
            ).all()

            # Convert to list of dictionaries
            result = []
            for model in models:
                model_path = resolve_path(model.model_path)
                result.append(
                    {
                        "id": model.id,
                        "model_type": model.model_type,
                        "version": model.version,
                        "model_path": model.model_path,
                        "accuracy": model.accuracy,
                        "created_at": (
                            model.created_at.isoformat() if model.created_at else None
                        ),
                        "is_active": model.is_active,
                        "file_exists": model_path.exists(),
                    }
                )

            return result

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.exception("Error getting models data: %s", str(e))
        return []


def main():
    """
    Main CLI entry point for model management.

    Parses command-line arguments and executes the appropriate action.
    """
    parser = argparse.ArgumentParser(
        description="Manage model versions for the bouldering analysis application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all models
  python src/manage_models.py list

  # List hold detection models only
  python src/manage_models.py list --model-type hold_detection

  # Activate a specific model version
  python src/manage_models.py activate --model-type hold_detection --version v1.0

  # Deactivate a specific model version
  python src/manage_models.py deactivate --model-type hold_detection --version v1.0
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    subparsers.required = True

    # List command
    list_parser = subparsers.add_parser("list", help="List model versions")
    list_parser.add_argument(
        "--model-type",
        type=str,
        help="Filter by model type (e.g., hold_detection, route_grading)",
    )

    # Activate command
    activate_parser = subparsers.add_parser("activate", help="Activate a model version")
    activate_parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        help="Model type (e.g., hold_detection, route_grading)",
    )
    activate_parser.add_argument(
        "--version", type=str, required=True, help="Model version to activate"
    )

    # Deactivate command
    deactivate_parser = subparsers.add_parser(
        "deactivate", help="Deactivate a model version"
    )
    deactivate_parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        help="Model type (e.g., hold_detection, route_grading)",
    )
    deactivate_parser.add_argument(
        "--version", type=str, required=True, help="Model version to deactivate"
    )

    args = parser.parse_args()

    # Execute the appropriate command
    if args.command == "list":
        result = list_models(args.model_type)
        print(result)
        sys.exit(0)

    elif args.command == "activate":
        success, message = activate_model(args.model_type, args.version)
        print(message)
        sys.exit(0 if success else 1)

    elif args.command == "deactivate":
        success, message = deactivate_model(args.model_type, args.version)
        print(message)
        sys.exit(0 if success else 1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
