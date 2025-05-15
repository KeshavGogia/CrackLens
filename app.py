from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from io import BytesIO
import base64

from preprocess import preprocess_image
from meta_ensemble import apply_metaensemble
from model_definitions import window_partition_tf
from crack_quantification import analyze_crack, plot_mask_and_graph_to_base64
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


# Model paths and loading
model_paths = {
    "unet": "models_new/unet_savedmodel",
    "attunet": "models_new/attunet_savedmodel",
    "swinunet": "models_new/swinunet_savedmodel",
    "transunet": "models_new/transunet_savedmodel",
    "raunet": "models_new/raunet_savedmodel"
}

custom_objects = {"window_partition_tf": window_partition_tf}
models = {name: load_model(path, compile=False, custom_objects=custom_objects) for name, path in model_paths.items()}
meta_model = load_model("models_new/meta_ensemble_savedmodel", compile=False)

def array_to_base64(img_array):
    img = Image.fromarray(img_array)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    try:
        # Process input image
        img_file = request.files['image']
        img = Image.open(img_file.stream).convert("RGB")
        preprocessed = preprocess_image(img)
        
        # Get predictions
        predictions = [model.predict(preprocessed) for model in models.values()]
        predictions_np = np.array(predictions).squeeze(axis=1)

        # Apply meta-ensemble
        final_output = apply_metaensemble(predictions_np, meta_model)
        mask = (final_output.squeeze() * 255).astype(np.uint8)

        # Perform crack analysis
        analysis = analyze_crack(mask)

        # Generate composite visualization (mask + graph)
        graph_vis_b64 = plot_mask_and_graph_to_base64(mask, analysis['graph'])

        return jsonify({
            'result_image': array_to_base64(mask),
            'skeleton_image': array_to_base64(analysis['skeleton']),
            'graph_image': graph_vis_b64,
            'analysis': {
                'length_pixels': analysis['length'],
                'width_stats': analysis['width_stats']
            }
        })

    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8000, debug=True)
