from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os
import gdown

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Step 1: Download models from Google Drive if not present ===
MODEL_PATHS = {
    "brain_classification": {
        "path": "Brain_Tumor_Classification_Model.h5",
        "url": "https://drive.google.com/uc?id=1HAcF3evhH8A4V_EnCWTVN0Gw9iU0VxnI"
    },
    "brain_segmentation": {
        "path": "brain_tumor_segmentation_model.h5",
        "url": "https://drive.google.com/uc?id=1mMaidey49WG1Kk4Evq2RFTTqXsKZxe2q"
    },
    "skin": {
        "path": "skin_cancer_model.h5",
        "url": "https://drive.google.com/uc?id=1MFc3BuuplJmm-Fbe637svgN7LNIjch8i"
    },
}

def download_models():
    for model in MODEL_PATHS.values():
        if not os.path.exists(model["path"]):
            print(f"Downloading {model['path']}...")
            gdown.download(model["url"], model["path"], quiet=False)

# === Step 2: Load models ===
def load_models():
    try:
        brain_classification_model = tf.keras.models.load_model(MODEL_PATHS['brain_classification']['path'])
        brain_segmentation_model = tf.keras.models.load_model(MODEL_PATHS['brain_segmentation']['path'])
        skin_model = tf.keras.models.load_model(MODEL_PATHS['skin']['path'])
        print("Models loaded successfully.")
        return brain_classification_model, brain_segmentation_model, skin_model
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None

# === Step 3: Preprocessing ===
def preprocess_image(image_bytes, img_size=(28, 28)):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize(img_size)
    image_array = np.array(image) / 255.0  # Normalize to [0,1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array.astype(np.float32)

# === Step 4: Routes ===
download_models()
brain_classification_model, brain_segmentation_model, skin_model = load_models()

@app.get("/health")
def health():
    return {
        "status": "OK",
        "models_loaded": {
            "brain_classification": brain_classification_model is not None,
            "brain_segmentation": brain_segmentation_model is not None,
            "skin": skin_model is not None
        }
    }

@app.post("/scan")
async def scan(
    image: UploadFile = File(...),
    diseaseType: str = Form(...)
):
    image_bytes = await image.read()
    results = {}

    if diseaseType == "Brain Tumor" and brain_classification_model and brain_segmentation_model:
        # أول حاجة نعمل Segmentation (بحجم 128x128)
        segmentation_input = preprocess_image(image_bytes, (128, 128))
        segmentation_output = brain_segmentation_model.predict(segmentation_input)
        mask = (segmentation_output[0, :, :, 0] * 255).astype(np.uint8)
        seg_image = Image.fromarray(mask, 'L')
        buf = io.BytesIO()
        seg_image.save(buf, format="PNG")
        encoded_mask = base64.b64encode(buf.getvalue()).decode('utf-8')
        results["segmentation_image_base64"] = encoded_mask

        # بعد كده Classification (بحجم 28x28)
        classification_input = preprocess_image(image_bytes, (28, 28))
        prediction = brain_classification_model.predict(classification_input)
        class_names = ["Glioma", "Meningioma", "No tumor", "Pituitary tumor"]
        idx = np.argmax(prediction)
        results["diagnosis"] = class_names[idx]
        results["confidence"] = f"{prediction[0][idx]*100:.2f}%"

    elif diseaseType == "Skin Cancer" and skin_model:
        processed_image = preprocess_image(image_bytes, (28, 28))
        prediction = skin_model.predict(processed_image)
        class_names = [
            "Actinic keratoses", "Basal cell carcinoma", "Benign keratosis-like lesions", 
            "Dermatofibroma", "Melanoma", "Melanocytic nevi", "Vascular lesions"
        ]
        idx = np.argmax(prediction)
        results["diagnosis"] = class_names[idx]
        results["confidence"] = f"{prediction[0][idx]*100:.2f}%"

    else:
        return JSONResponse(status_code=400, content={"error": "Invalid disease type or model not loaded"})

    print("Results:", results)
    return results
