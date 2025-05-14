from flask import Flask, request, jsonify
import os
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from io import BytesIO
import base64
from preprocess import preprocess_image
from meta_ensemble import apply_metaensemble

# Import your custom function
from model_definitions import window_partition_tf

app = Flask(__name__)

# Load all base models (SavedModel format)
model_paths = {
    "unet": "models_new/unet_savedmodel",
    "attunet": "models_new/attunet_savedmodel",
    "swinunet": "models_new/swinunet_savedmodel",
    "transunet": "models_new/transunet_savedmodel",
    "raunet": "models_new/raunet_savedmodel"
}

# Define custom objects for loading models with Lambda layers
custom_objects = {
    "window_partition_tf": window_partition_tf,
}

models = {
    name: load_model(path, compile=False, custom_objects=custom_objects)
    for name, path in model_paths.items()
}
meta_model = load_model("models_new/meta_ensemble_savedmodel", compile=False)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = request.files['image']
    img = Image.open(image.stream).convert("RGB")
    preprocessed = preprocess_image(img)
    print(f"Input shape: {preprocessed.shape}")
    predictions = [model.predict(preprocessed) for model in models.values()]
    predictions_np = np.array(predictions).squeeze(axis=1)  # (N_models, H, W, 1)

    final_output = apply_metaensemble(predictions_np, meta_model)

    # Convert output mask to base64
    final_img = Image.fromarray((final_output.squeeze() * 255).astype(np.uint8))
    buffered = BytesIO()
    final_img.save(buffered, format="PNG")
    encoded_img = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({'result_image': encoded_img})

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8000, debug=True)
