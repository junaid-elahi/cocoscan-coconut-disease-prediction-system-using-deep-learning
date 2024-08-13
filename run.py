# app.py

import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model
model = load_model('coconut_disease_final_model.h5')

# Define the class labels and fertilizer recommendations
class_labels = ['Bud Root Dropping', 'Bud Rot', 'Gray Leaf Spot', 'Leaf Rot', 'Stem Bleeding']
fertilizer_recommendations = {
    'Bud Root Dropping': 'Apply a balanced fertilizer with equal parts NPK (Nitrogen, Phosphorus, and Potassium).',
    'Bud Rot': 'Use fungicides and avoid waterlogging. Apply fertilizers high in Potassium.',
    'Gray Leaf Spot': 'Apply a balanced fertilizer with micronutrients like Zinc and Manganese.',
    'Leaf Rot': 'Use a balanced NPK fertilizer with added organic matter.',
    'Stem Bleeding': 'Apply Potassium-rich fertilizers and ensure proper drainage.'
}

def prepare_image(image_path):
    img = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        
        if file:
            filename = file.filename
            filepath = os.path.join('uploads', filename)
            file.save(filepath)

            # Prepare the image for prediction
            img_array = prepare_image(filepath)

            # Predict the class of the image
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            result = class_labels[predicted_class]
            fertilizer = fertilizer_recommendations[result]

            # Remove the uploaded file after prediction
            os.remove(filepath)

            return jsonify({'prediction': result, 'fertilizer': fertilizer})
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
