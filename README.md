
# Heart Disease Prediction Using XGBoost and Flask

## Project Overview
This project uses Machine Learning with the XGBoost algorithm to predict a heart disease class based on patient-related input features. A Flask API provides an endpoint for making predictions.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- XGBoost
- Flask
- Joblib

## Dataset
The project uses a heart disease dataset containing 303 records and 13 input features, with the target column `AHD`.

## Model Performance
- Algorithm: XGBoost Classifier
- Training records: 242
- Testing records: 61
- Test accuracy: 86.89%

The accuracy is based on one train-test split and may vary with different data.

## Project Features
- Data preprocessing
- Missing-value handling
- Feature scaling and categorical encoding
- XGBoost model training
- Model saving using Joblib
- Flask API for predictions

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train_model.py
```

### 3. Start the API
```bash
python app.py
```

### 4. Open in a browser
```text
http://127.0.0.1:5000
```

## Prediction Endpoint
- Method: POST
- Endpoint: `/predict`
- Content-Type: `application/json`

Send all 13 required feature fields as JSON.

## Disclaimer
This project is for educational purposes only. It is not intended for medical diagnosis or treatment decisions.
