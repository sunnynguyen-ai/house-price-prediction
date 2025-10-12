"""
Flask Web Application for House Price Prediction
Updated with config management and input validation
"""

from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import logging
from pathlib import Path

# Import configuration and validator
from config import (
    MODEL_PATH, 
    SCALER_PATH, 
    FLASK_CONFIG, 
    FEATURE_RANGES
)
from src.validator import InputValidator, FEATURE_DESCRIPTIONS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
model = None
scaler = None
validator = InputValidator(FEATURE_RANGES)

def load_model():
    """Load model with robust error handling"""
    global model, scaler
    
    try:
        if MODEL_PATH.exists() and SCALER_PATH.exists():
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            logger.info(f"Model loaded successfully from {MODEL_PATH}")
            return True
        else:
            logger.warning(f"Model files not found at {MODEL_PATH}")
            logger.warning("Please run 'python src/model_training.py' to train the model first.")
            return False
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

@app.route('/')
def home():
    """Render home page with prediction form"""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>House Price Predictor</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container { 
            max-width: 700px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        
        h1 { 
            text-align: center;
            color: #2d3748;
            margin-bottom: 10px;
            font-size: 2em;
        }
        
        .subtitle {
            text-align: center;
            color: #718096;
            margin-bottom: 30px;
            font-size: 0.95em;
        }
        
        .info { 
            background: #ebf8ff;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 25px;
            border-left: 4px solid #3182ce;
        }
        
        .info p {
            color: #2c5282;
            font-size: 0.9em;
            line-height: 1.5;
        }
        
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .form-group { 
            margin-bottom: 20px;
        }
        
        .form-group.full-width {
            grid-column: 1 / -1;
        }
        
        label { 
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #2d3748;
            font-size: 0.9em;
        }
        
        .label-description {
            font-weight: 400;
            color: #718096;
            font-size: 0.85em;
            margin-top: 2px;
        }
        
        input { 
            width: 100%;
            padding: 12px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 1em;
            transition: border-color 0.2s;
        }
        
        input:focus {
            outline: none;
            border-color: #667eea;
        }
        
        button { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 14px 28px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            width: 100%;
            margin-top: 10px;
            font-size: 1.1em;
            font-weight: 600;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        button:hover { 
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
        }
        
        button:active {
            transform: translateY(0);
        }
        
        .result { 
            margin-top: 25px;
            padding: 20px;
            background: #f0fff4;
            border-radius: 8px;
            border-left: 4px solid #48bb78;
            animation: slideIn 0.3s ease-out;
        }
        
        .result h3 {
            color: #22543d;
            font-size: 1.5em;
            margin-bottom: 8px;
        }
        
        .result p {
            color: #2f855a;
            font-size: 0.9em;
        }
        
        .error { 
            margin-top: 25px;
            padding: 20px;
            background: #fff5f5;
            border-radius: 8px;
            border-left: 4px solid #f56565;
            animation: slideIn 0.3s ease-out;
        }
        
        .error h3 {
            color: #742a2a;
            font-size: 1.2em;
            margin-bottom: 8px;
        }
        
        .error p {
            color: #c53030;
            font-size: 0.9em;
            margin-bottom: 5px;
        }
        
        .loading {
            display: none;
            text-align: center;
            margin-top: 20px;
        }
        
        .loading.active {
            display: block;
        }
        
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        @keyframes slideIn {
            from {
                opacity: 0;
                transform: translateY(-10px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 25px;
            }
            
            .form-grid {
                grid-template-columns: 1fr;
            }
            
            h1 {
                font-size: 1.5em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏠 House Price Predictor</h1>
        <p class="subtitle">California Housing Dataset</p>
        
        <div class="info">
            <p><strong>📊 How it works:</strong> Enter the housing features below to get an instant price prediction based on California housing data. Sample values are pre-filled for testing.</p>
        </div>
        
        <form id="predictionForm">
            <div class="form-grid">
                <div class="form-group">
                    <label>
                        Median Income
                        <div class="label-description">In $10,000s (e.g., 3.8 = $38,000)</div>
                    </label>
                    <input type="number" id="MedInc" step="0.1" value="3.8" required min="0.5" max="15">
                </div>
                
                <div class="form-group">
                    <label>
                        House Age
                        <div class="label-description">Years (1-52)</div>
                    </label>
                    <input type="number" id="HouseAge" value="28" required min="1" max="52">
                </div>
                
                <div class="form-group">
                    <label>
                        Average Rooms
                        <div class="label-description">Per household</div>
                    </label>
                    <input type="number" id="AveRooms" step="0.1" value="5.4" required min="1" max="20">
                </div>
                
                <div class="form-group">
                    <label>
                        Average Bedrooms
                        <div class="label-description">Per household</div>
                    </label>
                    <input type="number" id="AveBedrms" step="0.1" value="1.1" required min="0.5" max="10">
                </div>
                
                <div class="form-group">
                    <label>
                        Population
                        <div class="label-description">Block population</div>
                    </label>
                    <input type="number" id="Population" value="3000" required min="3" max="35682">
                </div>
                
                <div class="form-group">
                    <label>
                        Average Occupancy
                        <div class="label-description">Persons per household</div>
                    </label>
                    <input type="number" id="AveOccup" step="0.1" value="3.2" required min="0.5" max="20">
                </div>
                
                <div class="form-group">
                    <label>
                        Latitude
                        <div class="label-description">32.5 to 42.0</div>
                    </label>
                    <input type="number" id="Latitude" step="0.01" value="34.2" required min="32.5" max="42">
                </div>
                
                <div class="form-group">
                    <label>
                        Longitude
                        <div class="label-description">-124.5 to -114.0</div>
                    </label>
                    <input type="number" id="Longitude" step="0.01" value="-118.3" required min="-124.5" max="-114">
                </div>
            </div>
            
            <button type="submit">🔮 Predict Price</button>
        </form>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p style="margin-top: 10px; color: #718096;">Calculating prediction...</p>
        </div>
        
        <div id="result"></div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
            // Show loading
            document.getElementById('loading').classList.add('active');
            document.getElementById('result').innerHTML = '';
            
            const formData = {
                MedInc: parseFloat(document.getElementById('MedInc').value),
                HouseAge: parseFloat(document.getElementById('HouseAge').value),
                AveRooms: parseFloat(document.getElementById('AveRooms').value),
                AveBedrms: parseFloat(document.getElementById('AveBedrms').value),
                Population: parseFloat(document.getElementById('Population').value),
                AveOccup: parseFloat(document.getElementById('AveOccup').value),
                Latitude: parseFloat(document.getElementById('Latitude').value),
                Longitude: parseFloat(document.getElementById('Longitude').value)
            };
            
            fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(formData)
            })
            .then(response => response.json())
            .then(data => {
                // Hide loading
                document.getElementById('loading').classList.remove('active');
                
                const resultDiv = document.getElementById('result');
                if (data.success) {
                    resultDiv.innerHTML = `
                        <div class="result">
                            <h3>💰 Predicted Price: $${data.prediction.toLocaleString()}</h3>
                            <p>Based on the California housing dataset model</p>
                        </div>
                    `;
                } else {
                    let errorMessages = '<div class="error"><h3>⚠️ Validation Error</h3>';
                    if (Array.isArray(data.errors)) {
                        data.errors.forEach(err => {
                            errorMessages += `<p>• ${err}</p>`;
                        });
                    } else {
                        errorMessages += `<p>${data.error}</p>`;
                    }
                    errorMessages += '</div>';
                    resultDiv.innerHTML = errorMessages;
                }
            })
            .catch(error => {
                document.getElementById('loading').classList.remove('active');
                document.getElementById('result').innerHTML = 
                    '<div class="error"><h3>⚠️ Network Error</h3><p>Could not connect to server. Make sure the server is running.</p></div>';
            });
        });
    </script>
</body>
</html>
"""
    return html_content

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction with validation"""
    try:
        # Check if model is loaded
        if model is None or scaler is None:
            return jsonify({
                'success': False, 
                'error': 'Model not loaded. Please run "python src/model_training.py" to train the model first.'
            }), 503
        
        # Get JSON data
        data = request.json
        if not data:
            return jsonify({
                'success': False, 
                'error': 'No data provided'
            }), 400
        
        # Validate input
        is_valid, errors = validator.validate_features(data)
        if not is_valid:
            return jsonify({
                'success': False,
                'errors': errors
            }), 400
        
        # Extract features in correct order
        feature_order = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
                        'Population', 'AveOccup', 'Latitude', 'Longitude']
        features = [data[feature] for feature in feature_order]
        
        # Reshape for prediction
        features_array = np.array([features])
        
        # Scale and predict
        features_scaled = scaler.transform(features_array)
        prediction = model.predict(features_scaled)[0]
        
        # Ensure positive prediction
        prediction = max(prediction, 0)
        
        logger.info(f"Prediction made: ${prediction:,.2f}")
        
        return jsonify({
            'success': True,
            'prediction': round(prediction, 2)
        })
        
    except KeyError as e:
        logger.error(f"Missing feature: {e}")
        return jsonify({
            'success': False, 
            'error': f'Missing required feature: {str(e)}'
        }), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'success': False, 
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/status')
def status():
    """Check if model is loaded and get feature info"""
    return jsonify({
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'feature_ranges': validator.get_feature_info(),
        'feature_descriptions': FEATURE_DESCRIPTIONS
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_available': model is not None
    })

if __name__ == '__main__':
    print("="*60)
    print("HOUSE PRICE PREDICTION WEB APPLICATION")
    print("="*60)
    
    # Load model
    model_loaded = load_model()
    if model_loaded:
        print("✓ Model loaded successfully!")
    else:
        print("✗ Warning: Model not loaded")
        print("  Run 'python src/model_training.py' to train the model")
    
    print(f"\n🚀 Server starting at http://{FLASK_CONFIG['HOST']}:{FLASK_CONFIG['PORT']}")
    print("="*60)
    
    app.run(
        debug=FLASK_CONFIG['DEBUG'],
        host=FLASK_CONFIG['HOST'],
        port=FLASK_CONFIG['PORT']
    )
