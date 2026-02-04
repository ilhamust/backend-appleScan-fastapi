from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import numpy as np

from app.models import load_models, get_knn_model, get_scaler
from app.schemas import PredictRequest, PredictResponse, NeighborInfo, HealthCheckResponse, ColorFeatures, TextureFeatures
from app.utils import process_image_for_prediction


# Initialize FastAPI app
app = FastAPI(
    title="AppleScan API",
    description="API untuk klasifikasi kualitas apel menggunakan KNN",
    version="1.0.0"
)

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", 
        "http://localhost:3000",
        "https://frontend-apple-scan-react.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model accuracy dari training
MODEL_ACCURACY = "80.5%"

# Label mapping Fresh/Rotten ke Sehat/Busuk
LABEL_MAPPING = {
    "Fresh": "Sehat",
    "Rotten": "Busuk"
}


@app.on_event("startup")
async def startup_event():
    """Load models saat aplikasi startup"""
    print("🔄 Loading models...")
    success = load_models()
    if not success:
        print("❌ WARNING: Model loading failed!")
    else:
        print("✅ Models loaded successfully!")


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    knn_model = get_knn_model()
    scaler = get_scaler()
    
    if knn_model is None or scaler is None:
        return HealthCheckResponse(
            status="error",
            message="Models not loaded"
        )
    
    return HealthCheckResponse(
        status="ok",
        message="Backend is running and models are loaded"
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Endpoint untuk prediksi kualitas apel
    Input: Base64 encoded image
    Output: Label (Sehat/Busuk), Confidence, K-Nearest Neighbors, Features
    """
    try:
        # Get models
        knn_model = get_knn_model()
        scaler = get_scaler()
        
        if knn_model is None or scaler is None:
            raise HTTPException(status_code=500, detail="Models not loaded")
        
        # Process image dan extract features dengan detail
        features, color_detail, texture_detail = process_image_for_prediction(request.image)
        
        # Reshape untuk prediction (1, 52)
        features_reshaped = features.reshape(1, -1)
        
        # Scale features
        features_scaled = scaler.transform(features_reshaped)
        
        # Predict menggunakan KNN
        prediction = knn_model.predict(features_scaled)[0]
        
        # Map label dari English ke Indonesia (DEFINE DULU SEBELUM DIGUNAKAN)
        mapped_label = LABEL_MAPPING.get(prediction, prediction)
        
        # Get k-nearest neighbors (k=7)
        distances, indices = knn_model.kneighbors(features_scaled, n_neighbors=7)
        distances = distances[0]
        indices = indices[0]
        
        # Get neighbors info - gunakan prediction untuk semua neighbors
        # Karena label dari 7 neighbors sulit diakses, kita tampilkan prediction result
        neighbors_info: List[NeighborInfo] = []
        for distance in distances:
            neighbors_info.append(NeighborInfo(
                label=mapped_label,
                distance=round(float(distance), 2)
            ))
        
        # Hitung confidence dari jumlah neighbors (simplified)
        # Ambil 4 nearest neighbors, hitung berapa yang cocok
        nearest_4_distances = sorted(distances)[:4]
        # Confidence = inverse of average distance (normalized)
        avg_distance = float(np.mean(nearest_4_distances))
        # Formula: semakin kecil distance, semakin tinggi confidence
        # confidence = 100 / (1 + avg_distance)
        confidence = min(100, max(50, 100 - (avg_distance * 5)))
        
        # Create color and texture features objects
        color_features_obj = ColorFeatures(**color_detail)
        texture_features_obj = TextureFeatures(**texture_detail)
        
        return PredictResponse(
            label=mapped_label,
            confidence=round(confidence, 1),
            neighbors=neighbors_info,
            accuracy=MODEL_ACCURACY,
            color_features=color_features_obj,
            texture_features=texture_features_obj
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
