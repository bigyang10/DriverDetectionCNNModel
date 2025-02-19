#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import numpy as np
import tensorflow.lite as tflite
import requests
from PIL import Image
import os
import matplotlib.pyplot as plt
import cv2
import lime
import lime.lime_image
from skimage.segmentation import mark_boundaries

# Google Drive File ID (Replace with your own)
GDRIVE_FILE_ID = "1_76llMIxPd02byXinxWrHX_Je1nXWog2"

@st.cache_resource
def download_tflite_model():
    """Downloads the TFLite model from Google Drive if not already present."""
    model_path = "model.tflite"

    if not os.path.exists(model_path):  # Check if model is already downloaded
        st.write("Downloading model from Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
        response = requests.get(url, stream=True)

        with open(model_path, "wb") as f:
            f.write(response.content)
        st.write("✅ Model downloaded successfully!")
    else:
        st.write("✅ Model already exists. Loading...")

    # Load the TFLite model
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Load model from Google Drive
interpreter = download_tflite_model()

# Define class labels
CLASS_LABELS = {
    0: "c0 - Normal Driving",
    1: "c1 - Texting Right",
    2: "c2 - Talking on the Phone Right",
    3: "c3 - Texting Left",
    4: "c4 - Talking on the Phone Left",
    5: "c5 - Operating the Radio",
    6: "c6 - Drinking",
    7: "c7 - Reaching Behind",
    8: "c8 - Hair or Makeup",
    9: "c9 - Talking to Passenger"
}

# Preprocess image
def preprocess_image(image, input_shape):
    image = image.resize((input_shape[1], input_shape[2]))  # Resize to model input size
    image = np.array(image).astype(np.float32) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Make prediction
def predict(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    return output[0]  # Return probability scores

# LIME Explanation
def explain_with_lime(image, interpreter, class_index=0):
    """Generates a LIME heatmap for TFLite models."""
    explainer = lime.lime_image.LimeImageExplainer()
    
    # Convert image to numpy format for LIME
    image_np = np.array(image).astype(np.float32) / 255.0

    # Define a function to predict using TFLite
    def predict_fn(images):
        images = np.array(images, dtype=np.float32)
        predictions = []
        for img in images:
            img = np.expand_dims(img, axis=0)
            predictions.append(predict(interpreter, img))
        return np.array(predictions)

    # Generate LIME explanation
    explanation = explainer.explain_instance(
        image_np, 
        predict_fn, 
        top_labels=1, 
        hide_color=0, 
        num_samples=1000
    )

    # Get the heatmap
    temp, mask = explanation.get_image_and_mask(
        class_index, 
        positive_only=True, 
        num_features=5, 
        hide_rest=False
    )
    
    # Overlay the mask on the original image
    lime_heatmap = mark_boundaries(temp, mask)
    
    return lime_heatmap

# Streamlit UI
st.title("TFLite Image Classifier with LIME Explanation")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Get model input shape
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']

    # Preprocess the image
    processed_image = preprocess_image(image, input_shape)

    # Predict
    predictions = predict(interpreter, processed_image)

    # Get the class with the highest probability
    predicted_class = np.argmax(predictions)
    predicted_label = CLASS_LABELS[predicted_class]  # Get class label

    # Display results
    st.subheader("Prediction Result")
    st.write(f"**Predicted Class:** {predicted_label}")
    st.write(f"**Confidence Score:** {predictions[predicted_class]:.4f}")

    # Show probabilities as a bar chart
    st.subheader("Class Probabilities")
    fig, ax = plt.subplots()
    ax.bar(CLASS_LABELS.values(), predictions, color='lightblue')
    ax.set_xlabel("Class Name")
    ax.set_ylabel("Probability Score")
    ax.set_title("Model Prediction Confidence")
    ax.set_xticklabels(CLASS_LABELS.values(), rotation=45, ha='right', fontsize=9)
    st.pyplot(fig)

    # Apply LIME Explanation
    st.subheader("LIME Explanation (Heatmap)")
    lime_image = explain_with_lime(image, interpreter, predicted_class)
    st.image(lime_image, caption="LIME Heatmap", use_column_width=True)

    # Show probabilities for all classes
    for i, prob in enumerate(predictions):
        st.write(f"{CLASS_LABELS[i]}: {prob:.4f}")

