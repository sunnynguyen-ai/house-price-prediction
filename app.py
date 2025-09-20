"""
Flask Web Application for House Price Prediction - Simple Bug-Free Version
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Global variables
model = None
scaler = None

def load_model():
    """Load model with robust path checking"""
    global model, scaler
    
    # Check multiple possible locations
    paths_to_try = [
        ('models/house_price_model.joblib', 'models/scaler.joblib'),
        ('../models/house_price_model.joblib', '../models/scaler.joblib')
    ]
    
    for model_path, scaler_path in paths_to_try:
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            try:
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                print(f"Model loaded from: {model_path}")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                continue
    
    print("Model not found. Run 'python src/model_training.py' first.")
    return False

@app.route('/')
def home():
    """Simple home page"""
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>House Price Predictor</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; }
        .container { background: #f8f9fa; padding: 30px; border-radius: 8px; }
        h1 { text-align: center; color: #333; }
        .form-group { margin: 15px 0; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input { width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px; box-sizing: border-box; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; width: 100%; margin-top: 10px; }
        button:hover { background: #0056b3; }
        .result { margin-top: 20px; padding: 15px; background: #d4edda; border-radius: 4px; }
        .error { margin-top: 20px; padding: 15px; background: #f8d7da; border-radius: 4px; }
        .info { background: #e7f3ff; padding: 15px; border-radius: 4px; margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>House Price Predictor</h1>
        
        <div class="info">
            <p>Enter California housing features to predict price. Sample values are pre-filled.</p>
        </div>
        
        <form id="predictionForm">
            <div class="form-group">
                <label>Median Income (in $10,000s):</label>
                <input type="number" id="MedInc" step="0.1" value="3.8" required>
            </div>
            
            <div class="form-group">
                <label>House Age (years):</label>
                <input type="number" id="HouseAge" value="28" required>
            </div>
            
            <div class="form-group">
                <label>Average Rooms:</label>
                <input type="number" id="AveRooms" step="0.1" value="5.4" required>
            </div>
            
            <div class="form-group">
                <label>Average Bedrooms:</label>
                <input type="number" id="AveBedrms" step="0.1" value="1.1" required>
            </div>
            
            <div class="form-group">
                <label>Population:</label>
                <input type="number" id="Population" value="3000" required>
            </div>
            
            <div class="form-group">
                <label>Average Occupancy:</label>
                <input type="number" id="AveOccup" step="0.1" value="3.2" required>
            </div>
            
            <div class="form-group">
                <label>Latitude:</label>
                <input type="number" id="Latitude" step="0.01" value="34.2" required>
            </div>
            
            <div class="form-group">
                <label>Longitude:</label>
                <input type="number" id="Longitude" step="0.01" value="-118.3" required>
            </div>
            
            <button type="submit">Predict Price</button>
        </form>
        
        <div id="result"></div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', function(e) {
            e.preventDefault();
            
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
                const resultDiv = document.getElementById('result');
                if (data.success) {
                    resultDiv.innerHTML = '<div class="result"><h3>Predicted Price: $' + 
                        data.prediction.toLocaleString() + '</h3></div>';
                } else {
                    resultDiv.innerHTML = '<div class="error">Error: ' + data.error + '</div>';
                }
            })
            .catch(error => {
                document.getElementById('result').innerHTML = 
                    '<div class="error">Network error. Make sure the server is running.</div>';
            });
        });
    </script>
</body>
</html>
"""
    return html_content

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction"""
    try:
        if model is None or scaler is None:
            return jsonify({
                'success': False, 
                'error': 'Model not loaded. Run python src/model_training.py first.'
            })
        
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'})
        
        # Extract features in correct order
        features = [
            data.get('MedInc', 0),
            data.get('HouseAge', 0), 
            data.get('AveRooms', 0),
            data.get('AveBedrms', 0),
            data.get('Population', 0),
            data.get('AveOccup', 0),
            data.get('Latitude', 0),
            data.get('Longitude', 0)
        ]
        
        # Validate features
        for i, feature in enumerate(features):
            if not isinstance(feature, (int, float)) or feature == 0:
                return jsonify({'success': False, 'error': f'Invalid input for feature {i+1}'})
        
        # Reshape for prediction
        features_array = np.array([features])
        
        # Scale and predict
        features_scaled = scaler.transform(features_array)
        prediction = model.predict(features_scaled)[0]
        
        # Ensure positive prediction
        prediction = max(prediction, 0)
        
        return jsonify({
            'success': True,
            'prediction': round(prediction, 2)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/status')
def status():
    """Check if model is loaded"""
    return jsonify({
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })

if __name__ == '__main__':
    print("Starting House Price Prediction App...")
    
    # Load model
    model_loaded = load_model()
    if model_loaded:
        print("Model loaded successfully!")
    else:
        print("Warning: Model not loaded. Run 'python src/model_training.py' first.")
    
    print("Server starting at http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
