# preprocess.py
from PIL import Image
import numpy as np

def preprocess_image(img):
    # Convert to grayscale
    img = img.convert("L")
    
    # Resize to 256x256
    img = img.resize((256, 256))
    
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    
    # Add batch and channel dimensions (shape becomes [1, 256, 256, 1])
    return np.expand_dims(img_array, axis=(0, -1))
