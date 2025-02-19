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

# Simple Heatmap Generation
def generate_simple_heatmap(image):
    """Creates a basic heatmap overlay using random activation patterns."""
    image_np = np.array(image)
    heatmap = np.random.rand(image_np.shape[0], image_np.shape[1])  # Generate random heatmap
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)  # Apply blur to smooth out heatmap
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)  # Normalize to 0-255
    heatmap = np.uint8(heatmap)  # Convert to uint8
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Apply color mapping
    overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)  # Blend with original image
    return overlay

# Streamlit UI
st.title("TFLite Image Classifier with Simple Heatmap")

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

    # Apply Simple Heatmap Visualization
    st.subheader("Simple Heatmap Visualization")
    heatmap_image = generate_simple_heatmap(image)
    st.image(heatmap_image, caption="Simple Heatmap", use_column_width=True)

    # Show probabilities for all classes
    for i, prob in enumerate(predictions):
        st.write(f"{CLASS_LABELS[i]}: {prob:.4f}")

