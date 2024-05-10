from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import io
import base64
import re
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model(r"bestmodel.keras")

def preprocess_image(image):
    image = image.convert('L')
    image = image.resize((28, 28))
    image = np.array(image)
    image = image.reshape(1, 28, 28, 1)
    image = image.astype('float32')
    image = image / 255.0
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    img_data = request.form.get('image')
    
    if not img_data:
        return jsonify({'error': 'No image data provided'})

    img_str_match = re.search(r'base64,(.*)', img_data)
    
    if not img_str_match:
        return jsonify({'error': 'Invalid image data format'})
    
    img_str = img_str_match.group(1)
    image_bytes = io.BytesIO(base64.b64decode(img_str))
    image = Image.open(image_bytes)
    image = preprocess_image(image)
    predictions = model.predict(image)
    predicted_digit = np.argmax(predictions)
    return jsonify({'predicted_digit': str(predicted_digit)})

if __name__ == '__main__':
    app.run(debug=True)
