from pydantic import BaseModel
from typing import List


class PredictRequest(BaseModel):
    """Request model untuk endpoint predict"""
    image: str  # Base64 encoded image


class NeighborInfo(BaseModel):
    """Model untuk informasi k-nearest neighbor"""
    label: str
    distance: float


class ColorFeatures(BaseModel):
    """Model untuk color features"""
    h_histogram: List[float]  # 16 Hue histogram values
    s_histogram: List[float]  # 16 Saturation histogram values
    v_histogram: List[float]  # 16 Value histogram values


class TextureFeatures(BaseModel):
    """Model untuk texture features"""
    contrast: float
    homogeneity: float
    energy: float
    correlation: float


class PredictResponse(BaseModel):
    """Response model untuk endpoint predict"""
    label: str  # Sehat atau Busuk
    confidence: float  # Confidence dalam persen (0-100)
    neighbors: List[NeighborInfo]  # K-nearest neighbors
    accuracy: str  # Akurasi model training (80.5%)
    color_features: ColorFeatures  # Color histogram features
    texture_features: TextureFeatures  # Texture GLCM features


class HealthCheckResponse(BaseModel):
    """Response model untuk health check"""
    status: str
    message: str
