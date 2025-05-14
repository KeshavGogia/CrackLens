from PIL import Image
import numpy as np

def preprocess_image(img):
    img = img.convert("L")  
    img = img.resize((256, 256))
    arr = np.array(img) / 255.0  
    arr = np.expand_dims(arr, axis=(0, -1))  
    return arr.astype(np.float32)
