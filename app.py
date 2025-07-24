
import os
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import tensorflow as tf

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create upload directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
model = None

class PneumoniaPredictor:
    def __init__(self, model_path='models/pneumonia_vgg16_model.h5'):
        self.model_path = model_path
        self.model = None
        self.img_size = (224, 224)
        self.load_model()

    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                print(f"Model loaded successfully from {self.model_path}")
            else:
                print(f"Model file not found: {self.model_path}")
                print("Please train the model first using train_model.py")
        except Exception as e:
            print(f"Error loading model: {str(e)}")

    def preprocess_image(self, img_path):
        """Preprocess image for prediction"""
        try:
            # Load and resize image
            img = image.load_img(img_path, target_size=self.img_size)

            # Convert to array and normalize
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array / 255.0

            return img_array
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            return None

    def predict(self, img_path):
        """Make prediction on the image"""
        if self.model is None:
            return None, "Model not loaded"

        # Preprocess image
        processed_img = self.preprocess_image(img_path)
        if processed_img is None:
            return None, "Error processing image"

        try:
            # Make prediction
            prediction = self.model.predict(processed_img)[0][0]

            # Interpret prediction
            if prediction > 0.5:
                result = "PNEUMONIA"
                confidence = prediction * 100
            else:
                result = "NORMAL"
                confidence = (1 - prediction) * 100

            return result, confidence
        except Exception as e:
            return None, f"Error making prediction: {str(e)}"

# Initialize predictor
predictor = PneumoniaPredictor()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Make prediction
            result, confidence = predictor.predict(file_path)

            if result is None:
                return jsonify({'error': confidence})

            # Prepare response
            response = {
                'prediction': result,
                'confidence': f"{confidence:.2f}%",
                'image_path': f"static/uploads/{filename}",
                'filename': filename
            }

            return render_template('result.html', **response)

        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'})

    return jsonify({'error': 'Invalid file type'})

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    if file and allowed_file(file.filename):
        try:
            # Save uploaded file
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Make prediction
            result, confidence = predictor.predict(file_path)

            if result is None:
                return jsonify({'error': confidence})

            # Clean up uploaded file (optional)
            # os.remove(file_path)

            return jsonify({
                'prediction': result,
                'confidence': f"{confidence:.2f}%",
                'status': 'success'
            })

        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'})

    return jsonify({'error': 'Invalid file type'})

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle server errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("Starting Pneumonia Detection Web Application...")
    print("Make sure you have trained the model using train_model.py")
    print("Access the application at: http://localhost:5000")

    app.run(debug=True, host='0.0.0.0', port=5000)
