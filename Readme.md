# 🧠 CrackLens: Intelligent Crack Detection & Quantification

CrackLens is an AI-powered system that enables automated detection and quantification of cracks in infrastructure using deep learning. It provides visual segmentation of cracks from images and computes dimensional metrics to help in structural health monitoring and preventive maintenance.

---

## 🚀 Features

- 📷 Upload an image of a structural surface (wall, road, beam, etc.)
- 🧠 Runs multiple segmentation models: UNet, Attention UNet, RAUNet, TransUNet, SwinUNet
- 🤖 Uses a Meta-Ensembler model to aggregate predictions from all models for enhanced accuracy
- 📐 Calculates crack metrics (length, width) for quantification
- 💻 Web interface to interact with the model
- 🛠️ Fully modular backend using Flask
- 🌐 Frontend for seamless interaction 

---

## 🧩 Model Architecture

- `UNet`
- `Attention UNet`
- `RAUNet`
- `TransUNet`
- `SwinUNet`
- `MetaEnsembler` – Combines all models' predictions into a single optimized output.

Each model is pre-trained and saved as a `.h5` file (converted to TensorFlow SavedModel format for compatibility).

---
## 🔄 Model Conversion

Some models were originally saved in `.h5` format, which caused compatibility issues with newer TensorFlow versions (e.g., deserialization errors due to unrecognized arguments like `batch_shape`).

To ensure smooth execution and compatibility with TensorFlow 2.13+, all `.h5` models are converted to **TensorFlow SavedModel** format using a utility script.

#### 🛠️ How to Convert

Run the following command from the project root:

```bash
python convert_models.py
```
---
## 🖥️ Backend (Flask)

- Accepts image upload via API
- Applies preprocessing consistent with training
- Loads all segmentation models
- Generates individual predictions and feeds them to a meta-ensembler
- Returns final segmented crack image + crack size estimations
- Runs on a configurable port (default: `http://localhost:8000`)

### API Endpoints

- `POST /predict`  
   Accepts an image and returns the ensembled segmentation mask along with crack measurement data.

---

## 🌐 Frontend

- Simple, responsive UI
- Users can:
  - Upload an image
  - View segmentation results
  - See quantitative metrics like crack width, length, etc.
- Fetches predictions via Flask backend
- Displays output overlayed on original image

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/CrackLens.git

cd CrackLens
```
### 2. Backend setup

```bash
cd backend

conda create -p venv python=3.8

conda activate venv

pip install -r requirements.txt

python app.py
```
