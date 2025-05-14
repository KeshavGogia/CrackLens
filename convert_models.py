import os
import h5py
import tensorflow as tf
from model_definitions import build_unet, build_attention_unet, build_raunet, build_swin_unet, build_transunet, build_metaensemble

model_builders = {
    "unet": build_unet,
    "attunet": build_attention_unet,
    "swinunet": build_swin_unet,
    "transunet": build_transunet,
    "raunet": build_raunet,
    "meta_ensemble": build_metaensemble
}

h5_model_paths = {
    "unet": "models/final_unet.h5",
    "attunet": "models/final_attunet.h5",
    "swinunet": "models/final_swinunet.h5",
    "transunet": "models/final_transunet.h5",
    "raunet": "models/final_raunet.h5",
    "meta_ensemble": "models/meta_ensemble.h5"
}

input_shapes = {
    "unet": (256, 256, 1),
    "attunet": (256, 256, 1),
    "swinunet": (256, 256, 1),
    "transunet": (256, 256, 1),
    "raunet": (256, 256, 1),
    "meta_ensemble": (5,)  
}

output_base = "models_new"

for name, h5_path in h5_model_paths.items():
    print(f"Converting {name}...")

    model_fn = model_builders[name]
    model = model_fn(input_shape=input_shapes[name])

    model.load_weights(h5_path)
    
    saved_model_dir = os.path.join(output_base, f"{name}_savedmodel")
    model.save(saved_model_dir)
    print(f"✅ Saved {name} to {saved_model_dir}")

print("🏁 All models converted successfully!")
