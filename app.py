import os
import joblib
from flask import Flask, render_template, request, jsonify
from src.prediction import predict_stroke

app = Flask(__name__)

MODEL_PATH = 'models/random_forest_model.joblib'
SCALER_PATH = 'models/scaler.joblib'
ENCODERS_PATH = 'models/encoders.joblib'
BMI_MEDIAN_PATH = 'models/bmi_median.joblib'
FEAT_NAMES_PATH = 'models/feature_names.joblib'

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    bmi_median = joblib.load(BMI_MEDIAN_PATH)
    feature_names = joblib.load(FEAT_NAMES_PATH)
    print("Successfully loaded ML artifacts.")
except Exception as e:
    print(f"Error loading model files: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        result = predict_stroke(model, data, encoders = encoders, bmi_median = bmi_median, scaler = scaler, feature_names = feature_names)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(debug = False, host = '0.0.0.0', port = port)