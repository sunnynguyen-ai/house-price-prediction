"""
Configuration management for house price prediction project
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
NOTEBOOKS_DIR = PROJECT_ROOT / 'notebooks'

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, NOTEBOOKS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model configuration
MODEL_CONFIG = {
    'random_forest': {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'random_state': 42,
        'n_jobs': -1
    }
}

# Data configuration
DATA_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'validation_size': 0.1
}

# Feature validation ranges (for input validation)
FEATURE_RANGES = {
    'MedInc': (0.5, 15.0),      # Median income in $10,000s
    'HouseAge': (1, 52),         # House age in years
    'AveRooms': (1, 20),         # Average rooms per household
    'AveBedrms': (0.5, 10),      # Average bedrooms per household
    'Population': (3, 35682),    # Block population
    'AveOccup': (0.5, 20),       # Average occupancy
    'Latitude': (32.5, 42.0),    # California latitude range
    'Longitude': (-124.5, -114.0) # California longitude range
}

# Flask configuration
FLASK_CONFIG = {
    'DEBUG': os.getenv('FLASK_DEBUG', 'True').lower() == 'true',
    'HOST': os.getenv('FLASK_HOST', '0.0.0.0'),
    'PORT': int(os.getenv('FLASK_PORT', 5000))
}

# Model file paths
MODEL_PATH = MODELS_DIR / 'house_price_model.joblib'
SCALER_PATH = MODELS_DIR / 'scaler.joblib'
METRICS_PATH = MODELS_DIR / 'metrics.json'
