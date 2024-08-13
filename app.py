import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import streamlit as st
import os

# Load the model
model = tf.keras.models.load_model('model.h5')

# Define the class labels (adjust as per your classes)
class_labels = ['Bud Root Dropping', 'Bud Rot', 'Gray Leaf Spot', 'Leaf Rot', 'Stem Bleeding']

def prepare_image(image):
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Rescale image
    return image

# Streamlit app
st.title("Coconut Disease Prediction App")
st.write("Upload an image to predict the disease.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Save uploaded file to a directory
    os.makedirs('./uploads', exist_ok=True)
    file_path = os.path.join('./uploads', uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())
    
    # Load and prepare the image
    image = load_img(file_path)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    image = prepare_image(image)
    predictions = model.predict(image)

    # Make prediction
    predictions = model.predict(image)
    predicted_class = class_labels[np.argmax(predictions)]

    st.write(f"Predicted Class: {predicted_class}")