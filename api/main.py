"""
FastAPI Server for Pediatric Chest X-Ray Pneumonia Detection

"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import uvicorn
import os
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI with metadata
app = FastAPI(
    title="🏥 PneumoDetectAI - Pediatric Pneumonia Detection API",
    description="Clinical-grade AI pneumonia screening: 86% cross-operator validation accuracy, 96.4% sensitivity (485 samples)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
model_info = {
    "loaded": False,
    "load_time": None,
    "model_path": None,
    "performance": {
        "accuracy": 86.0,
        "sensitivity": 96.4,
        "specificity": 74.8,
        "false_positive_rate": 25.2,
        "roc_auc": 0.964,
        "pr_auc": 0.968
    }
}

@app.on_event("startup")
async def load_model():
    """Load the trained model on startup"""
    global model, model_info
    try:
        
        model_paths = [
            Path("../models/best_chest_xray_model.h5"), 
            Path("models/best_chest_xray_model.h5"),
            Path("./best_chest_xray_model.h5"),
            Path("streamlit_api_folder/best_chest_xray_model.h5")
        ]
       
        for model_path in model_paths:
            if model_path.exists():
                logger.info(f"Loading model from: {model_path}")
                model = tf.keras.models.load_model(model_path)
                model_info.update({
                    "loaded": True,
                    "load_time": datetime.now().isoformat(),
                    "model_path": str(model_path)
                })
                logger.info("✅ Model loaded successfully!")
                break
        else:
            logger.error("❌ Model file not found in any expected location")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess image for model prediction
    Matches the preprocessing used during training
    """
    try:
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        # Resize to model input size (224x224 for MobileNetV2)
        image = image.resize((224, 224))
        # Convert to numpy array
        img_array = np.array(image)
        # Normalize pixel values to [0, 1] (same as training)
        img_array = img_array.astype(np.float32) / 255.0
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image preprocessing failed: {str(e)}")

def interpret_prediction(prediction_score: float) -> dict:
    """
    Interpret model prediction with confidence levels
    Based on cross-operator validation performance metrics
    """
    #  model uses 0.5 as threshold (>0.5 = pneumonia, <=0.5 = normal)
    if prediction_score > 0.5:
        diagnosis = "PNEUMONIA"
        confidence = float(prediction_score * 100)
        # Confidence levels based on cross-operator validation performance
        if confidence >= 80:
            confidence_level = "High"
            recommendation = "Strong indication of pneumonia. Recommend immediate medical attention."
        elif confidence >= 60:
            confidence_level = "Moderate"
            recommendation = "Moderate indication of pneumonia. Medical review recommended."
        else:
            confidence_level = "Low"
            recommendation = "Possible pneumonia detected. Further examination advised."
    else:
        diagnosis = "NORMAL"
        confidence = float((1 - prediction_score) * 100)
        if confidence >= 80:
            confidence_level = "High"
            recommendation = "No signs of pneumonia detected. Chest X-ray appears normal."
        elif confidence >= 60:
            confidence_level = "Moderate"
            recommendation = "Likely normal chest X-ray. Routine follow-up if symptoms persist."
        else:
            confidence_level = "Low"
            recommendation = "Unclear result. Manual review by radiologist recommended."
    return {
        "diagnosis": diagnosis,
        "confidence": round(confidence, 2),
        "confidence_level": confidence_level,
        "recommendation": recommendation,
        "raw_score": float(prediction_score)
    }

# API Routes
@app.get("/")
def read_root():
    """Root endpoint with API information"""
    return {
        "message": "🏥 PneumoDetectAI pneumonia detection API",
        "status": "running",
        "model_loaded": model_info["loaded"],
        "performance": model_info["performance"],
        "version": "1.0.0",
        "description": "AI-powered pneumonia detection with 86% cross-operator validation accuracy",
        "endpoints": {
            "predict": "/predict - Upload chest X-ray for analysis",
            "health": "/health - Check API health status",
            "info": "/info - Get detailed model information",
            "docs": "/docs - Interactive API documentation"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model_info["loaded"] else "unhealthy",
        "model_loaded": model_info["loaded"],
        "load_time": model_info["load_time"],
        "timestamp": datetime.now().isoformat(),
        "performance_summary": "86% accuracy, 96.4% sensitivity, 25.2% false positive rate"
    }

@app.get("/info")
def model_info_endpoint():
    """Detailed model information"""
    return {
        "model_info": model_info,
        "clinical_validation": {
            "accuracy": "86.0%",
            "sensitivity": "96.4%",
            "specificity": "74.8%",
            "clinical_readiness": "READY for clinical validation"
        },
        "cross_operator_validation": {
            "dataset_size": "485 independent samples",
            "normal_cases": "234",
            "pneumonia_cases": "251",
            "generalization": "Good (8.8% drop from internal validation)"
        },
        "technical_specs": {
            "architecture": "MobileNetV2 with custom classification head",
            "input_size": "224x224 RGB images",
            "training_data": "Balanced dataset (1:1 ratio)",
            "preprocessing": "Resize to 224x224, normalize to [0,1]"
        },
        "usage_guidelines": {
            "intended_use": "Preliminary pneumonia screening assistant",
            "limitations": "Not a replacement for professional diagnosis",
            "recommendation": "Always consult healthcare professionals for medical decisions"
        }
    }

@app.post("/predict")
async def predict_pneumonia(file: UploadFile = File(...)):
    """
    Predict pneumonia from chest X-ray image
    Upload a chest X-ray image to get AI-powered pneumonia detection
    with 86% cross-operator validation accuracy and 96.4% sensitivity.
    """
    # Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs and restart."
        )
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG, etc.)"
        )
    # Check file size (limit to 10MB)
    if hasattr(file, 'size') and file.size > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File size too large. Maximum 10MB allowed."
        )
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        # Log image info
        logger.info(f"Processing image: {image.size}, mode: {image.mode}")
        # Preprocess image
        processed_image = preprocess_image(image)
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)[0]
        # Interpret results
        result = interpret_prediction(prediction)
        # Add metadata with cross-operator validation performance
        result.update({
            "timestamp": datetime.now().isoformat(),
            "filename": file.filename,
            "image_size": f"{image.size}x{image.size[1]}",
            "cross_operator_validation_performance": {
                "accuracy": "86.0%",
                "sensitivity": "96.4%",
                "specificity": "74.8%",
                "validated_on": "485 independent samples"
            },
            "disclaimer": "This AI assistant is for preliminary screening only. Always consult healthcare professionals for medical decisions."
        })
        logger.info(f"Prediction completed: {result['diagnosis']} ({result['confidence']:.1f}%)")
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/stats")
def get_model_stats():
    """Get model performance statistics from cross-operator validation"""
    return {
        "performance_metrics": {
            "overall_accuracy": "86.0%",
            "sensitivity": "96.4%",
            "specificity": "74.8%",
            "precision": "80.4%",
            "false_positive_rate": "25.2%",
            "false_negative_rate": "3.6%",
            "roc_auc": "0.964",
            "pr_auc": "0.968"
        },
        "cross_operator_validation_confusion_matrix": {
            "true_negatives": 175,
            "false_positives": 59,
            "false_negatives": 9,
            "true_positives": 242,
            "total_test_samples": 485
        },
        "clinical_interpretation": {
            "excellent_screening": "96.4% sensitivity ideal for pneumonia screening",
            "false_alarm_consideration": "25.2% false positive rate requires clinical review",
            "high_detection_rate": "96.4% of pneumonia cases correctly identified",
            "clinical_readiness": "Ready for real-world clinical validation"
        },
        "validation_methodology": {
            "type": "cross_operator_validation",
            "dataset": "485 independent samples",
            "generalization": "Good (8.8% drop from internal validation)"
        }
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found. Visit /docs for API documentation."}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again or contact support."}
    )

if __name__ == "__main__":
   
    uvicorn.run(
        "api.main:app", 
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 7860)),  
        reload=False  
    )
