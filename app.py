
from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("heart_disease_model.joblib")

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Heart Disease Prediction API",
        "status": "running",
        "usage": "Send POST request to /predict"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "JSON input is required"}), 400

        input_df = pd.DataFrame([data])

        expected_columns = [
            "Age", "Sex", "ChestPain", "RestBP", "Chol",
            "Fbs", "RestECG", "MaxHR", "ExAng", "Oldpeak",
            "Slope", "Ca", "Thal"
        ]

        missing = [
            col for col in expected_columns
            if col not in input_df.columns
        ]

        if missing:
            return jsonify({
                "error": "Missing required fields",
                "missing_fields": missing
            }), 400

        input_df = input_df[expected_columns]

        prediction = int(model.predict(input_df)[0])

        return jsonify({
            "prediction": prediction,
            "result": (
                "Higher-risk class predicted by the model"
                if prediction == 1
                else "Lower-risk class predicted by the model"
            ),
            "note": "Educational demonstration only; not a medical diagnosis."
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
