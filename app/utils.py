import cv2
import numpy as np
import base64
from io import BytesIO
from PIL import Image
from skimage.feature import graycomatrix, graycoprops


IMG_SIZE = 100


def decode_base64_image(image_base64: str) -> np.ndarray:
    """Decode base64 string menjadi numpy array (OpenCV format BGR)"""
    try:
        # Hapus header data:image/png;base64, jika ada
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        
        # Decode base64
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data))
        
        # Convert ke BGR format (OpenCV)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return image_cv
        
    except Exception as e:
        raise ValueError(f"Error decoding image: {str(e)}")


def resize_image(image: np.ndarray, size: int = IMG_SIZE) -> np.ndarray:
    """Resize image ke ukuran yang ditentukan"""
    return cv2.resize(image, (size, size))


def extract_color_features(image: np.ndarray) -> tuple:
    """Ekstraksi fitur warna menggunakan histogram HSV (48 features + detail)"""
    # Convert BGR ke HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Hitung histogram untuk setiap channel
    h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])
    s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256])
    v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256])
    
    # Normalize individual histograms
    h_hist = h_hist.flatten() / np.sum(h_hist)
    s_hist = s_hist.flatten() / np.sum(s_hist)
    v_hist = v_hist.flatten() / np.sum(v_hist)
    
    # Gabungkan untuk model
    hist = np.concatenate([h_hist, s_hist, v_hist])
    
    return hist, {
        'h_histogram': h_hist.tolist(),
        's_histogram': s_hist.tolist(),
        'v_histogram': v_hist.tolist()
    }


def extract_texture_features(image: np.ndarray) -> tuple:
    """Ekstraksi fitur tekstur menggunakan GLCM (4 features + detail)"""
    # Convert ke grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Hitung GLCM
    glcm = graycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
    
    # Extract properties
    contrast = float(graycoprops(glcm, 'contrast')[0, 0])
    homogeneity = float(graycoprops(glcm, 'homogeneity')[0, 0])
    energy = float(graycoprops(glcm, 'energy')[0, 0])
    correlation = float(graycoprops(glcm, 'correlation')[0, 0])
    
    features_array = np.array([contrast, homogeneity, energy, correlation])
    
    return features_array, {
        'contrast': round(contrast, 4),
        'homogeneity': round(homogeneity, 4),
        'energy': round(energy, 4),
        'correlation': round(correlation, 4)
    }


def extract_all_features(image: np.ndarray) -> tuple:
    """Ekstraksi semua fitur (color + texture = 52 features) dengan detail"""
    color_array, color_detail = extract_color_features(image)
    texture_array, texture_detail = extract_texture_features(image)
    
    # Gabungkan untuk model: 48 color features + 4 texture features = 52 total
    all_features = np.concatenate([color_array, texture_array])
    
    return all_features, color_detail, texture_detail


def process_image_for_prediction(image_base64: str) -> tuple:
    """Pipeline lengkap: decode -> resize -> extract features (dengan detail)"""
    try:
        # Decode base64
        image = decode_base64_image(image_base64)
        
        # Resize
        image = resize_image(image, IMG_SIZE)
        
        # Extract features dengan detail
        features, color_detail, texture_detail = extract_all_features(image)
        
        return features, color_detail, texture_detail
        
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")
