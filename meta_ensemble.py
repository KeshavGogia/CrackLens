import numpy as np

def apply_metaensemble(model_outputs, meta_model):
    """
    model_outputs: np.array of shape (N_models, H, W, 1)
    meta_model: a trained meta-ensemble model
    """
    n_models, H, W, _ = model_outputs.shape
    stacked = model_outputs.transpose(1, 2, 0, 3).reshape(H, W, n_models)  # (H, W, N_models)
    
    # Flatten for model
    input_meta = stacked.reshape(-1, n_models)  # (H*W, N_models)

    # Predict
    meta_preds = meta_model.predict(input_meta)
    final_output = meta_preds.reshape(H, W, 1)
    return final_output
