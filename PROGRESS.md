# Project Roadmap

To create a bouldering route analysis tool using user snapshots and a pre-trained model, we'll need to approach this as a computer vision and machine learning project. Here's a proposed plan:

## Revised Plan: Bouldering Route Analysis Tool (Python-focused)

### Phase 1: Enhanced Data Management and Preprocessing

This is the most critical and labor-intensive phase, as the quality and quantity of your data will directly impact the model's performance.

1. **Dataset Organization**:
   - **Strategy**: Collect a large and diverse set of bouldering route images. This could involve:
     - Taking photos at various climbing gyms/outdoor locations.
     - Searching online for publicly available bouldering images (ensure licensing allows for use).
   - **Leverage Existing Structure**: Utilize existing `data/sample_hold/` and `data/sample_route/` directories for initial testing and development.
   - **Diversity**: Ensure images vary in lighting, wall color, hold types, angles, and route complexity. Aim for hundreds to thousands of images.
   - **Data Versioning**: Implement DVC (Data Version Control) for tracking dataset changes and maintaining reproducibility.

2. **Data Annotation (Manual Classification)**:
   - **Tools**: Use an image annotation tool (e.g., LabelImg, VGG Image Annotator (VIA), RectLabel, or even custom scripts with libraries like OpenCV) to manually annotate each image.
   - **Hold Identification**: For each image, draw bounding boxes around every hold that is part of a route. For each bounding box, classify the hold type (e.g., "crimp", "jug", "sloper", "pinch", "pocket", "foot-hold", "start-hold", "top-out-hold"). This will be the dataset for object detection.
   - **Route Grading**: For each *entire route* in an image, assign a V-grade (e.g., V0, V1, ..., V10+). This will be the target for the route grading model. You might need to define clear criteria for grading consistency.
   - **Output Format**: Annotations should be saved in a format compatible with machine learning frameworks (e.g., Pascal VOC XML, COCO JSON, or YOLO TXT format).

3. **Data Preprocessing and Augmentation**:
   - **Splitting**: Divide the annotated dataset into training, validation, and test sets (e.g., 70/15/15%).
   - **Augmentation**: Apply transformations like rotation, flipping, scaling, brightness/contrast adjustments to increase the effective size and diversity of the dataset, making the model more robust. Python libraries like `Albumentations` or `imgaug` are excellent for this.

4. **Active Learning Pipeline**:
   - **Database Schema**: Create database schema for storing user feedback and corrected grades.
   - **Feedback Integration**: Implement mechanism for incorporating user-corrected grades into training data.
   - **Quality Control**: Add validation checks for user-submitted corrections to ensure data quality.

### Phase 2: Improved Model Architecture

1. **Hold Detection Model**:
   - **Leverage Existing Model**: Fine-tune the existing `yolov8n.pt` model with bouldering-specific data.
   - **Architecture**: Use YOLOv8 from Ultralytics (already selected as the preferred model).
   - **Framework**: PyTorch (recommended for YOLO models).
   - **Training**: Train the chosen model on your annotated hold dataset. This will involve fine-tuning a pre-trained model (e.g., trained on COCO dataset) on your specific bouldering hold dataset.
   - **Model Versioning**: Implement model versioning for A/B testing and performance tracking.
   - **Confidence Thresholding**: Add confidence thresholding for reliable predictions.

2. **Route Grading Model**:
   - **Feature Extraction**: Extract features from detected holds:
     - Hold count and type distribution
     - Spatial arrangement analysis
     - Difficulty indicators (small holds, dynamic moves, etc.)
   - **Approach**: Use feature extraction + regression/classification:
     - Extract features from the detected holds (e.g., number of holds, types of holds, spatial arrangement, average hold size).
     - Feed these features into a separate machine learning model (e.g., a Random Forest Regressor, Support Vector Regressor, or a simple Neural Network) to predict the V-grade.
   - **Uncertainty Quantification**: Add uncertainty estimation for predictions to provide confidence intervals.
   - **Model Versioning**: Track different versions of grading models for comparison.

### Phase 3: Enhanced Web Application Development

1. **Backend Improvements** (`src/main.py`):
   - **Leverage Existing Structure**: Enhance the existing Flask application.
   - **RESTful API Endpoints**:
     - POST `/analyze` - Image upload and analysis
     - POST `/feedback` - User grade correction
     - GET `/stats` - Usage statistics and model performance
     - GET `/health` - Health check endpoint
   - **Image Preprocessing**: Implement server-side image loading and preprocessing (resizing, normalization) using `Pillow` or `OpenCV`.
   - **Model Inference**: Load the trained hold detection and route grading models. Pass the preprocessed image through the hold detection model, extract features, and then pass them to the route grading model.
   - **Result Generation**: Format the results (e.g., JSON containing detected hold coordinates, types, and the predicted route grade).
   - **Database Integration**: Integrate SQLite or PostgreSQL for storing user feedback and analysis results.
   - **Asynchronous Processing**: Implement async processing for heavy computations to improve user experience.

2. **Frontend Enhancements** (`src/templates/index.html`):
   - **Leverage Existing Structure**: Enhance the existing HTML template.
   - **Interactive UI**: 
     - Image upload form with drag-and-drop support
     - Display area for the uploaded image with hold overlays
     - Interactive grade correction interface
     - Real-time feedback visualization
   - **User Experience**:
     - Clean, intuitive, and uncluttered design
     - Mobile-responsive layout
     - Loading indicators during processing
     - Error handling and user feedback
   - **JavaScript Integration**: Use JavaScript to handle image uploads, send requests to the backend API, and display the results dynamically.

3. **Testing Framework** (`tests/`):
   - **Unit Tests**: Implement tests for individual components
   - **Integration Tests**: Test API endpoints and data flow
   - **Model Tests**: Validate model performance on test datasets

### Phase 4: Continuous Learning System

1. **Feedback Integration**:
   - **Automated Retraining**: Implement pipeline for retraining models with new user-corrected data
   - **Performance Monitoring**: Track model accuracy over time
   - **Quality Control**: Validate new data before incorporation into training set

2. **Deployment and Monitoring**:
   - **Containerization**: Package the Python application and its dependencies into a Docker container (leverage existing `setup_env.sh`).
   - **CI/CD Pipeline**: Implement automated testing and deployment.
   - **Cloud Deployment**: Deploy the Docker container to a cloud platform (e.g., AWS EC2/ECS, Google Cloud Run/App Engine, Azure App Service) or a private server.
   - **Performance Monitoring**: Monitor application performance and model accuracy with alerts for degradation.

### Phase 5: Documentation and Maintenance

1. **Documentation**:
   - **Expand README**: Enhance the minimal `src/README.md` with comprehensive documentation.
   - **API Documentation**: Document all API endpoints with examples.
   - **User Guide**: Create user guide for the web application.

2. **Maintenance**:
   - **Regular Updates**: Keep dependencies and models up to date.
   - **Performance Optimization**: Continuously optimize model inference and application performance.
   - **Security**: Implement security best practices for user data and API endpoints.

## Key Python Libraries to Consider:

- **Image Processing**: `Pillow`, `OpenCV-Python`
- **Machine Learning Frameworks**: `PyTorch` (recommended for YOLO models) or `TensorFlow`
- **Object Detection**: `ultralytics/yolov8` (already selected)
- **Data Augmentation**: `Albumentations`, `imgaug`
- **Web Framework**: `Flask` (already selected for simplicity and quick prototyping)
- **Database**: `SQLAlchemy` (for ORM), `psycopg2-binary` (for PostgreSQL)
- **Data Versioning**: `dvc`
- **Testing**: `pytest`, `pytest-cov`
- **API Documentation**: `flask-restx` or `flask-swagger-ui`
- **Asynchronous Processing**: `celery` (for background tasks)

## Hardware Considerations:

- **Training**: The availability of an **RTX 3080** is excellent for model training. This GPU will significantly accelerate the training process for both the hold detection and route grading models.
- **Inference**: The GTX 1060 can serve as a backup or for testing inference on a less powerful machine.

## User Interface Design:

For the user interface, we will focus on a **clean, intuitive, and uncluttered design**. This means:

- A clear section for image upload.
- A prominent display area for the uploaded image and the analysis results (detected holds with bounding boxes/labels, and the predicted grade).
- Minimal navigation or extraneous elements to keep the focus on the core functionality.
- We'll use standard web technologies (HTML, CSS, JavaScript) for the frontend to ensure broad compatibility and a straightforward implementation.

## Implementation Roadmap:

1. **Week 1-2**: Set up development environment, database schema, and basic API endpoints
2. **Week 3-4**: Implement hold detection model fine-tuning and basic analysis endpoint
3. **Week 5-6**: Develop route grading model and feature extraction pipeline
4. **Week 7-8**: Build frontend interface with image upload and results display
5. **Week 9-10**: Implement user feedback system and continuous learning pipeline
6. **Week 11-12**: Testing, documentation, and deployment preparation

The overall plan maintains the solid foundation of the original roadmap while addressing gaps in user feedback integration, continuous learning, and leveraging the existing codebase more effectively.
