# app.py
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# --- FIX START ---
# st.set_page_config() MUST be the very first Streamlit command in the script.
st.set_page_config(page_title="SolarGuard: Solar Panel Condition Classifier", layout="wide")
# --- FIX END ---

# --- Configuration ---
CLASSIFICATION_IMG_HEIGHT, CLASSIFICATION_IMG_WIDTH = 128, 128
CLASSIFICATION_MODEL_PATH = 'solar_panel_classifier_model.h5'

# --- Load the Trained Model ---
@st.cache_resource
def load_classification_model(path):
    if not os.path.exists(path):
        st.error(f"Error: Classification model file not found at '{path}'. Please ensure 'model_training.py' was run successfully.")
        st.stop()
    model = tf.keras.models.load_model(path)
    return model

# Load model when the app starts
classification_model = load_classification_model(CLASSIFICATION_MODEL_PATH)

# --- Define Class Names (MUST match the order from training) ---
classification_class_names = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-Damage', 'Snow-Covered']


# --- Streamlit App Interface ---
# All subsequent st. commands will work fine after set_page_config()

st.title("☀️ SolarGuard: Intelligent Defect Detection on Solar Panels")
st.markdown("---")

# --- Project Objectives Section (Classification Task Only) ---
st.header("Project Objectives")

st.subheader("1. Classification Task: Identifying Solar Panel Conditions") 
st.markdown("**Aim:** Develop a classification model to categorize solar panel images into one of six conditions: Clean, Dusty, Bird-Drop, Electrical-Damage, Physical-Damage, and Snow-Covered.") 
st.markdown("**Use Case:**") 
st.markdown("- Automate the process of identifying the condition of solar panels from images. ")
st.markdown("- Provide insights into common issues affecting panel efficiency. ")
st.markdown("- Help solar maintenance teams prioritize cleaning and repair tasks. ")
st.markdown("**Possible Inputs (Features):** Raw solar panel images.") 
st.markdown("**Target:** A category label indicating the panel condition.") 

st.markdown("---")

# --- Interactive Prediction Section ---
st.header("Predict Solar Panel Condition")
st.markdown("Upload an image of a solar panel below to classify its condition.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    original_image_pil = Image.open(uploaded_file).convert('RGB')
    st.image(original_image_pil, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Analyzing image...")

    # --- Classification Prediction ---
    st.markdown("### Classification Task Result:")
    st.markdown("---")
    
    st.markdown("**Possible Input (Features):**") 
    st.markdown(f"- Raw solar panel image (Uploaded: `{uploaded_file.name}`)") 

    classification_image = original_image_pil.resize((CLASSIFICATION_IMG_HEIGHT, CLASSIFICATION_IMG_WIDTH))
    classification_img_array = np.array(classification_image) / 255.0
    classification_img_array = np.expand_dims(classification_img_array, axis=0)

    predictions = classification_model.predict(classification_img_array, verbose=0)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = classification_class_names[predicted_class_index]
    confidence = np.max(predictions[0]) * 100

    # This directly displays the "category label indicating the panel condition" as requested.
    st.markdown("**Target:** A category label indicating the panel condition.") 
    st.markdown(f"- **Predicted Condition:** **{predicted_class_name}**")
    st.markdown(f"- **Confidence:** {confidence:.2f}%")

    st.success(f"The model has classified the panel condition as **'{predicted_class_name}'** with {confidence:.2f}% confidence.")
    
    st.markdown("---")
    st.markdown("""
    **Project Status:**
    This application currently focuses on **Solar Panel Classification**, identifying the overall condition of the panel.
    * **Classification:** Implemented using a Deep Learning (MobileNetV2) model for automated condition identification.
    * **Future Enhancements:** The project can be extended to include **Object Detection** (to pinpoint specific defects on the panel), requiring extensive manual data annotation and specialized model training.
    """)

else:
    st.info("Please upload an image to get a classification.")