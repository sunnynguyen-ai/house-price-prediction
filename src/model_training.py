"""
House Price Prediction Model Training Pipeline
Updated with config management, cross-validation, and improved logging
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json
import logging
from datetime import datetime
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import (
    MODEL_CONFIG,
    DATA_CONFIG,
    MODEL_PATH,
    SCALER_PATH,
    METRICS_PATH,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HousePricePredictor:
    """Complete pipeline for house price prediction model training"""
    
    def __init__(self, config=None):
        """
        Initialize the predictor with configuration
        
        Args:
            config: Optional model configuration dictionary
        """
        if config is None:
            config = MODEL_CONFIG['random_forest']
        
        self.model = RandomForestRegressor(**config)
        self.scaler = StandardScaler()
        self.feature_names = None
        self.metrics = {}
        self.training_date = datetime.now().isoformat()
        
        logger.info("Initialized HousePricePredictor")
    
    def load_data(self):
        """
        Load California housing dataset
        
        Returns:
            DataFrame with features and target price
        """
        logger.info("Loading California housing dataset...")
        
        try:
            housing = fetch_california_housing()
            
            # Create DataFrame
            df = pd.DataFrame(housing.data, columns=housing.feature_names)
            df['price'] = housing.target * 100000  # Convert to actual price range
            
            self.feature_names = housing.feature_names.tolist()
            
            logger.info(f"Dataset loaded successfully")
            logger.info(f"  - Samples: {df.shape[0]:,}")
            logger.info(f"  - Features: {df.shape[1]-1}")
            logger.info(f"  - Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
            logger.info(f"  - Mean price: ${df['price'].mean():,.0f}")
            logger.info(f"  - Median price: ${df['price'].median():,.0f}")
            
            # Save raw data
            self._save_raw_data(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _save_raw_data(self, df):
        """Save raw data to CSV"""
        try:
            RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
            output_path = RAW_DATA_DIR / 'california_housing.csv'
            df.to_csv(output_path, index=False)
            logger.info(f"Raw data saved to {output_path}")
        except Exception as e:
            logger.warning(f"Could not save raw data: {e}")
    
    def explore_data(self, df):
        """
        Perform basic data exploration
        
        Args:
            df: Input DataFrame
        """
        logger.info("\n" + "="*70)
        logger.info("DATA EXPLORATION")
        logger.info("="*70)
        
        # Check for missing values
        missing = df.isnull().sum().sum()
        logger.info(f"Missing values: {missing}")
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        logger.info(f"Duplicate rows: {duplicates}")
        
        # Basic statistics
        logger.info("\nFeature Statistics:")
        logger.info("-" * 70)
        for col in self.feature_names:
            logger.info(f"{col:<15} - Mean: {df[col].mean():>10.2f}, Std: {df[col].std():>10.2f}")
        
        # Correlations with price
        logger.info("\nCorrelation with Price:")
        logger.info("-" * 70)
        correlations = df[self.feature_names].corrwith(df['price']).sort_values(ascending=False)
        for feature, corr in correlations.items():
            logger.info(f"{feature:<15} - {corr:>6.3f}")
    
    def preprocess_data(self, df):
        """
        Prepare data for modeling
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("\n" + "="*70)
        logger.info("DATA PREPROCESSING")
        logger.info("="*70)
        
        # Features and target
        X = df[self.feature_names]
        y = df['price']
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=DATA_CONFIG['test_size'], 
            random_state=DATA_CONFIG['random_state']
        )
        
        logger.info(f"Training set size: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
        logger.info(f"Test set size: {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")
        
        # Scale features
        logger.info("Scaling features with StandardScaler...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Save processed data
        self._save_processed_data(X_train_scaled, X_test_scaled, y_train, y_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def _save_processed_data(self, X_train, X_test, y_train, y_test):
        """Save processed data"""
        try:
            PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            
            # Save as numpy arrays for faster loading
            np.save(PROCESSED_DATA_DIR / 'X_train.npy', X_train)
            np.save(PROCESSED_DATA_DIR / 'X_test.npy', X_test)
            np.save(PROCESSED_DATA_DIR / 'y_train.npy', y_train)
            np.save(PROCESSED_DATA_DIR / 'y_test.npy', y_test)
            
            logger.info(f"Processed data saved to {PROCESSED_DATA_DIR}")
        except Exception as e:
            logger.warning(f"Could not save processed data: {e}")
    
    def train_model(self, X_train, y_train, use_cross_validation=True):
        """
        Train the Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training target
            use_cross_validation: Whether to perform cross-validation
        """
        logger.info("\n" + "="*70)
        logger.info("MODEL TRAINING")
        logger.info("="*70)
        
        logger.info(f"Model: {self.model.__class__.__name__}")
        logger.info(f"Parameters: {self.model.get_params()}")
        
        # Cross-validation before training
        if use_cross_validation:
            logger.info("\nPerforming 5-fold cross-validation...")
            cv_scores = cross_val_score(
                self.model, X_train, y_train, 
                cv=5, 
                scoring='r2',
                n_jobs=-1
            )
            logger.info(f"Cross-validation R² scores: {cv_scores}")
            logger.info(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            
            self.metrics['cv_r2_mean'] = float(cv_scores.mean())
            self.metrics['cv_r2_std'] = float(cv_scores.std())
        
        # Train model
        logger.info("\nTraining model on full training set...")
        self.model.fit(X_train, y_train)
        logger.info("✓ Model training completed!")
        
        # Training set performance
        train_predictions = self.model.predict(X_train)
        train_r2 = r2_score(y_train, train_predictions)
        logger.info(f"Training R² score: {train_r2:.4f}")
        
        self.metrics['train_r2'] = float(train_r2)
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate model performance on test set
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("\n" + "="*70)
        logger.info("MODEL EVALUATION")
        logger.info("="*70)
        
        predictions = self.model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        
        # Store metrics
        self.metrics.update({
            'test_mae': float(mae),
            'test_rmse': float(rmse),
            'test_r2': float(r2),
            'test_mape': float(mape),
            'n_samples_train': int(len(X_test) / DATA_CONFIG['test_size'] * (1 - DATA_CONFIG['test_size'])),
            'n_samples_test': int(len(X_test)),
            'training_date': self.training_date
        })
        
        # Display results
        logger.info(f"Mean Absolute Error (MAE):        ${mae:,.2f}")
        logger.info(f"Root Mean Square Error (RMSE):    ${rmse:,.2f}")
        logger.info(f"R² Score:                         {r2:.4f} ({r2*100:.2f}%)")
        logger.info(f"Mean Absolute Percentage Error:   {mape:.2f}%")
        
        # Performance interpretation
        logger.info("\n" + "-"*70)
        if r2 >= 0.8:
            logger.info("✓ Excellent model performance!")
        elif r2 >= 0.7:
            logger.info("✓ Good model performance")
        elif r2 >= 0.6:
            logger.info("○ Acceptable model performance")
        else:
            logger.info("✗ Model performance needs improvement")
        logger.info("-"*70)
        
        return self.metrics
    
    def feature_importance(self):
        """Display and return feature importance"""
        logger.info("\n" + "="*70)
        logger.info("FEATURE IMPORTANCE")
        logger.info("="*70)
        
        if self.feature_names is None:
            logger.warning("Feature names not available")
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Display
        for idx, row in importance_df.iterrows():
            bar_length = int(row['importance'] * 50)
            bar = '█' * bar_length
            logger.info(f"{row['feature']:<15} {bar} {row['importance']:.4f}")
        
        # Store top features
        self.metrics['top_features'] = importance_df.head(5).to_dict('records')
        
        return importance_df
    
    def save_model(self):
        """Save trained model, scaler, and metrics"""
        logger.info("\n" + "="*70)
        logger.info("SAVING MODEL")
        logger.info("="*70)
        
        try:
            # Ensure models directory exists
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            
            # Save model
            joblib.dump(self.model, MODEL_PATH)
            logger.info(f"✓ Model saved to: {MODEL_PATH}")
            
            # Save scaler
            joblib.dump(self.scaler, SCALER_PATH)
            logger.info(f"✓ Scaler saved to: {SCALER_PATH}")
            
            # Save metrics
            with open(METRICS_PATH, 'w') as f:
                json.dump(self.metrics, f, indent=2)
            logger.info(f"✓ Metrics saved to: {METRICS_PATH}")
            
            # Display file sizes
            model_size = MODEL_PATH.stat().st_size / 1024 / 1024
            scaler_size = SCALER_PATH.stat().st_size / 1024
            logger.info(f"\nModel size: {model_size:.2f} MB")
            logger.info(f"Scaler size: {scaler_size:.2f} KB")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_trained_model(self):
        """Load a previously trained model"""
        try:
            self.model = joblib.load(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            
            if METRICS_PATH.exists():
                with open(METRICS_PATH, 'r') as f:
                    self.metrics = json.load(f)
            
            logger.info("✓ Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False


def main():
    """Main training pipeline"""
    print("\n" + "="*70)
    print("HOUSE PRICE PREDICTION - MODEL TRAINING PIPELINE")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    try:
        # Initialize predictor
        predictor = HousePricePredictor()
        
        # Load and explore data
        df = predictor.load_data()
        predictor.explore_data(df)
        
        # Preprocess data
        X_train, X_test, y_train, y_test = predictor.preprocess_data(df)
        
        # Train model
        predictor.train_model(X_train, y_train, use_cross_validation=True)
        
        # Evaluate performance
        metrics = predictor.evaluate_model(X_test, y_test)
        
        # Show feature importance
        predictor.feature_importance()
        
        # Save model
        success = predictor.save_model()
        
        if success:
            print("\n" + "="*70)
            print("✓ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*70)
            print(f"\nYou can now run the Flask app:")
            print(f"  python app.py")
            print(f"\nModel files saved in: {MODEL_PATH.parent}")
            print("="*70)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
