"""
Data loading and preprocessing for California housing dataset
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import os

def load_california_housing():
    """Load California housing dataset from sklearn"""
    housing = fetch_california_housing()
    
    # Create DataFrame
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['target'] = housing.target
    
    return df

def save_raw_data():
    """Save raw data to data/raw folder"""
    df = load_california_housing()
    os.makedirs('../data/raw', exist_ok=True)
    df.to_csv('../data/raw/california_housing.csv', index=False)
    print("Raw data saved to data/raw/california_housing.csv")

if __name__ == "__main__":
    save_raw_data()
