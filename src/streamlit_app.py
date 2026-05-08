import os
os.environ["KERAS_BACKEND"] = "torch" # Force Torch backend for GPU/Grad-CAM consistency

import streamlit as st
import sys

# Ensure src directory is in path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import keras
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from interpretability import make_gradcam_heatmap

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Rice AI Diagnostics",
    page_icon="🌾",
    layout="wide"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #28a745;
        color: white;
    }
    .status-box {
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        background-color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🌾 Rice Leaf Disease Diagnostic System")
st.write("Professional Vision Pipeline for Intelligent Agriculture")

# ==========================================
# CACHED MODEL LOADING
# ==========================================

@st.cache_resource
def load_rice_model():
    model_path = os.path.join(os.path.dirname(__file__), "../models/rice_ai_best.keras")
    if not os.path.exists(model_path):
        # Fallback to current dir if not in models folder
        model_path = "rice_leaf_disease_model.keras"
    return keras.models.load_model(model_path)

try:
    model = load_rice_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ==========================================
# CONFIGURATION
# ==========================================

classes = ["Bacterialblight", "Blast", "Brownspot", "Tungro"]
IMG_SIZE = 224

def preprocess_image(image):
    image = image.convert("RGB") # Ensure 3 channels even if source is RGBA
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    return image

def overlay_heatmap(image, heatmap, alpha=0.4):
    # Convert PIL to numpy
    img = np.array(image.resize((IMG_SIZE, IMG_SIZE)))
    
    # Rescale heatmap to 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Resize to original image size
    jet_heatmap = cv2.resize(jet_heatmap, (img.shape[1], img.shape[0]))
    jet_heatmap = np.uint8(255 * jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    return superimposed_img.astype("uint8")

# ==========================================
# LAYOUT
# ==========================================

col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("📁 Image Ingestion")
    uploaded_file = st.file_uploader(
        "Upload a clear rice leaf image (JPG/PNG)",
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Source Image", use_container_width=True)

with col2:
    if uploaded_file:
        st.subheader("🔬 AI Diagnostic Analysis")
        
        with st.spinner("Analyzing image patterns..."):
            # 1. Preprocess and Predict
            img_array = preprocess_image(image)
            preds = model.predict(img_array, verbose=0)
            pred_idx = np.argmax(preds[0])
            confidence = preds[0][pred_idx]
            
            # 2. Results Header
            st.markdown(f"""
                <div class='status-box'>
                    <h3>Result: {classes[pred_idx]}</h3>
                    <p>Focal Confidence: {confidence*100:.2f}%</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.write("") # Spacer

            # 3. Explainability (Grad-CAM)
            try:
                heatmap = make_gradcam_heatmap(img_array, model)
                overlay = overlay_heatmap(image, heatmap)
                
                st.subheader("🎯 Visual Focus (Grad-CAM)")
                st.write("The AI focused on the highlighted areas to make its decision.")
                st.image(overlay, use_container_width=True)
            except Exception as ge:
                st.warning(f"Grad-CAM visualization unavailable: {ge}")

            # 4. Probabilities Chart
            st.subheader("📊 Class Probabilities")
            chart_data = {cls: float(prob) for cls, prob in zip(classes, preds[0])}
            st.bar_chart(chart_data)

    else:
        st.info("Waiting for image upload to begin analysis...")

# ==========================================
# FOOTER
# ==========================================
st.divider()
st.caption("AI Rice Leaf Diagnostic Pipeline | Keras 3 Multi-Backend Implementation")