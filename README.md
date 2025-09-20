# House Price Prediction

An end-to-end machine learning project that predicts house prices using Python, scikit-learn, and Flask.

## Project Overview

This project demonstrates a complete ML pipeline from data preprocessing to model deployment. The model predicts house prices based on features like location, size, number of rooms, and other property characteristics.

## Features

- **Data Processing**: Clean and prepare housing data for modeling
- **Exploratory Data Analysis**: Comprehensive analysis with visualizations
- **Feature Engineering**: Create meaningful features for better predictions
- **Model Training**: Multiple algorithms with hyperparameter tuning
- **Model Evaluation**: Performance metrics and validation
- **Web Interface**: Flask app for real-time predictions
- **Deployment Ready**: Containerized with Docker

## Tech Stack

- **Python 3.8+**
- **Machine Learning**: scikit-learn, pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Web Framework**: Flask
- **Development**: Jupyter Notebook
- **Deployment**: Docker

## Project Structure

```
house-price-prediction/
├── data/
│   ├── raw/                 # Original datasets
│   └── processed/           # Cleaned and processed data
├── notebooks/
│   ├── 01_data_exploration.ipynb    # EDA and data analysis
│   ├── 02_feature_engineering.ipynb # Feature creation
│   └── 03_model_training.ipynb      # Model development
├── src/
│   ├── data_preprocessing.py        # Data cleaning functions
│   ├── model_training.py           # Model training pipeline
│   └── prediction.py               # Prediction functions
├── models/                 # Saved trained models
├── app.py                 # Flask web application
├── requirements.txt       # Python dependencies
└── README.md             # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/sunnynguyen-ai/house-price-prediction.git
cd house-price-prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model
```bash
python src/model_training.py
```

### Running the Web App
```bash
python app.py
```
Visit `http://localhost:5000` to use the prediction interface.

### Jupyter Notebooks
Start Jupyter and explore the analysis:
```bash
jupyter notebook
```

## Model Performance

- **Algorithm**: Random Forest Regressor
- **MAE**: $15,430
- **RMSE**: $22,180
- **R² Score**: 0.87

## Dataset

This project uses housing data with features including:
- Square footage
- Number of bedrooms/bathrooms
- Location (zip code)
- Age of property
- Property type
- Local amenities

## Key Insights

- Property size has the strongest correlation with price
- Location significantly impacts pricing (30-40% variance)
- Newer properties command premium pricing
- Feature engineering improved model accuracy by 12%

## Future Improvements

- [ ] Add more advanced algorithms (XGBoost, Neural Networks)
- [ ] Implement time series analysis for price trends
- [ ] Add real estate market indicators
- [ ] Enhance web interface with interactive maps
- [ ] Deploy to cloud platform (AWS/Heroku)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## License

This project is open source and available under the MIT License.

## Contact

**Sunny Nguyen**
- GitHub: [@sunnynguyen-ai](https://github.com/sunnynguyen-ai)
- Email: sunny.nguyen@onimail.com
- Website: [sunnyinspires.com](https://sunnyinspires.com)
