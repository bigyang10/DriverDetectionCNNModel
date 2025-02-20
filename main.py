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
    image = np.array(image).astype(np.float32) / 255.0         # Normalize
    image = np.expand_dims(image, axis=0)                        # Add batch dimension
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

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state.history = []

st.title("TFLite Image Classifier with History")

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

    st.subheader("Class Probabilities")
    fig = create_bar_chart(predictions)
    st.pyplot(fig)

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
    # Iterate in reverse order so the most recent upload is first
    for i, entry in enumerate(reversed(st.session_state.history), start=1):
        # Only display entries that meet the confidence threshold
        if entry["confidence_score"] >= confidence_threshold:
            st.markdown(f"### History Entry {i}")
            st.image(entry["image"], caption="Uploaded Image", use_column_width=True)
            st.write(f"**Predicted Class:** {entry['predicted_label']}")
            st.write(f"**Confidence Score:** {entry['confidence_score']:.4f}")
            hist_fig = create_bar_chart(entry["predictions"])
            st.pyplot(hist_fig)

