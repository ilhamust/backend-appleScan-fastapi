import pickle
import os
from pathlib import Path

# Path ke folder models
MODEL_DIR = Path(__file__).parent.parent / "models"

# Global variables untuk menyimpan model dan scaler
knn_model = None
scaler = None


def load_models():
    """Load model KNN dan scaler dari file pkl"""
    global knn_model, scaler
    
    try:
        model_path = MODEL_DIR / "model_knn_apel.pkl"
        scaler_path = MODEL_DIR / "scaler.pkl"
        
        # Cek apakah file ada
        if not model_path.exists():
            raise FileNotFoundError(f"Model file tidak ditemukan: {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file tidak ditemukan: {scaler_path}")
        
        # Load model KNN
        with open(model_path, 'rb') as f:
            knn_model = pickle.load(f)
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        print("✓ Model KNN dan Scaler berhasil dimuat")
        return True
        
    except Exception as e:
        print(f"✗ Error loading models: {str(e)}")
        return False


def get_knn_model():
    """Get KNN model"""
    return knn_model


def get_scaler():
    """Get scaler"""
    return scaler
