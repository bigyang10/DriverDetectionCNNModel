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
GDRIVE_FILE_ID = "1vQqLq9Ho5mpB_tY2_pEvV0tGJMys3zl5"

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

# Generate a bar chart figure from predictions
def create_bar_chart(predictions):
    fig, ax = plt.subplots()
    ax.bar(list(CLASS_LABELS.values()), predictions, color='lightblue')
    ax.set_xlabel("Class Name")
    ax.set_ylabel("Probability Score")
    ax.set_title("Model Prediction Confidence")
    ax.set_xticklabels(list(CLASS_LABELS.values()), rotation=45, ha='right', fontsize=9)
    return fig

# Generate Class Activation Map (CAM)
def generate_cam(image, interpreter, predicted_class):
    """Generates CAM for the predicted class."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get feature maps from the last convolutional layer
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    feature_maps = interpreter.get_tensor(input_details[0]['index'])
    feature_maps = np.squeeze(feature_maps)

    # Get the weights for the predicted class
    class_weights = interpreter.get_tensor(output_details[0]['index'])
    class_weights = np.squeeze(class_weights)[predicted_class]

    # Weighted sum of feature maps
    cam = np.zeros(feature_maps.shape[:2], dtype=np.float32)
    for i, w in enumerate(class_weights):
        cam += w * feature_maps[:, :, i]

    # Normalize and apply color map
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min())
    cam = cv2.resize(cam, (image.shape[1], image.shape[0]))
    cam = np.uint8(255 * cam)
    cam = cv2.applyColorMap(cam, cv2.COLORMAP_JET)

    # Overlay the CAM on the original image
    overlay = cv2.addWeighted(np.array(image), 0.6, cam, 0.4, 0)
    return overlay

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state.history = []

st.title("TFLite Image Classifier with CAM & History")

# Sidebar slider for filtering history entries by confidence score
confidence_threshold = st.sidebar.slider("Minimum Confidence to Display History", 0.0, 1.0, 0.0, 0.01)

# File uploader for new image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Get model input shape and preprocess the image
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    processed_image = preprocess_image(image, input_shape)

    # Predict and retrieve the predicted class and confidence score
    predictions = predict(interpreter, processed_image)
    predicted_class = np.argmax(predictions)
    predicted_label = CLASS_LABELS[predicted_class]
    confidence_score = predictions[predicted_class]

    # Display current prediction results
    st.subheader("Prediction Result")
    st.write(f"**Predicted Class:** {predicted_label}")
    st.write(f"**Confidence Score:** {confidence_score:.4f}")

    # Use Tabs for Bar Chart and CAM
    st.subheader("Visualization Options")
    tab1, tab2 = st.tabs(["Bar Chart", "Class Activation Map (CAM)"])

    with tab1:
        st.subheader("Class Probabilities")
        fig = create_bar_chart(predictions)
        st.pyplot(fig)

    with tab2:
        st.subheader("Class Activation Map (CAM)")
        cam_image = generate_cam(processed_image, interpreter, predicted_class)
        st.image(cam_image, caption="Class Activation Map (CAM)", use_column_width=True)

    # Save the current result to history
    st.session_state.history.append({
        "image": image,
        "predicted_label": predicted_label,
        "confidence_score": confidence_score,
        "predictions": predictions
    })

# Display history entries
if st.session_state.history:
    st.subheader("History")
    for i, entry in enumerate(reversed(st.session_state.history), start=1):
        if entry["confidence_score"] >= confidence_threshold:
            st.markdown(f"### History Entry {i}")
            st.image(entry["image"], caption="Uploaded Image", use_column_width=True)
            st.write(f"**Predicted Class:** {entry['predicted_label']}")
            st.write(f"**Confidence Score:** {entry['confidence_score']:.4f}")
            hist_fig = create_bar_chart(entry["predictions"])
            st.pyplot(hist_fig)

