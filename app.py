import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the pre-trained model
MODEL_PATH = 'model/model.h5'
model = load_model(MODEL_PATH)

# Image preprocessing function
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# Streamlit UI
st.title("Crack Detection Model")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    img_path = f"images/{uploaded_file.name}"
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(img_path, caption="Uploaded Image.", use_column_width=True)
    
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    label = "Crack Detected" if prediction[0][0] > 0.5 else "No Crack Detected"
    
    st.write(f"Prediction: {label}")
