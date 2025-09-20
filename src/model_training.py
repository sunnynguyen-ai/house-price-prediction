"""
House Price Prediction Model Training - Working Implementation
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os

class HousePricePredictor:
    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=10,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def load_data(self):
        """Load California housing dataset"""
        print("Loading California housing dataset...")
        housing = fetch_california_housing()
        
        # Create DataFrame
        df = pd.DataFrame(housing.data, columns=housing.feature_names)
        df['price'] = housing.target * 100000  # Convert to actual price range
        
        self.feature_names = housing.feature_names
        
        print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]-1} features")
        print(f"Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}")
        
        return df
    
    def preprocess_data(self, df):
        """Prepare data for modeling"""
        print("Preprocessing data...")
        
        # Features and target
        X = df[self.feature_names]
        y = df['price']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """Train the Random Forest model"""
        print("Training Random Forest model...")
        self.model.fit(X_train, y_train)
        print("Model training completed!")
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        print("Evaluating model performance...")
        predictions = self.model.predict(X_test)
        
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        
        print("\n" + "="*50)
        print("MODEL PERFORMANCE RESULTS")
        print("="*50)
        print(f"Mean Absolute Error (MAE): ${mae:,.2f}")
        print(f"Root Mean Square Error (RMSE): ${rmse:,.2f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Accuracy: {r2*100:.2f}%")
        print("="*50)
        
        return {'mae': mae, 'rmse': rmse, 'r2': r2}
    
    def feature_importance(self):
        """Display feature importance"""
        if self.feature_names is not None:
            importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nFEATURE IMPORTANCE:")
            print("-" * 30)
            for _, row in importance.iterrows():
                print(f"{row['feature']:<15}: {row['importance']:.4f}")
    
    def save_model(self, model_path='../models/house_price_model.joblib', 
                   scaler_path='../models/scaler.joblib'):
        """Save trained model and scaler"""
        os.makedirs('../models', exist_ok=True)
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        print(f"\nModel saved to: {model_path}")
        print(f"Scaler saved to: {scaler_path}")

def main():
    """Main training pipeline"""
    print("HOUSE PRICE PREDICTION MODEL TRAINING")
    print("=" * 50)
    
    # Initialize predictor
    predictor = HousePricePredictor()
    
    # Load and preprocess data
    df = predictor.load_data()
    X_train, X_test, y_train, y_test = predictor.preprocess_data(df)
    
    # Train model
    predictor.train_model(X_train, y_train)
    
    # Evaluate performance
    metrics = predictor.evaluate_model(X_test, y_test)
    
    # Show feature importance
    predictor.feature_importance()
    
    # Save model
    predictor.save_model()
    
    print("\nTraining pipeline completed successfully!")
    return metrics

if __name__ == "__main__":
    main()
