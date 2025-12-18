import os
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.append(os.path.dirname(__file__))
from src.preprocess import preprocessing
from src.training import train_models, save_model
from src.prediction import predict_stroke, sample_patient, sample_patient_raw

def main():
    data = preprocessing(filepath='data/healthcare-dataset-stroke-data.csv')
    if data is None:
        print("\nERROR: Data preprocessing failed!")
        return

    results = train_models(
        data['X_train'],
        data['y_train'],
        data['X_test'],
        data['y_test'],
        use_smote = True
    )

    os.makedirs('models', exist_ok = True)
    
    best_model_name = results['best_model']
    best_model = results['models'][best_model_name]
    
    save_model(best_model, f'models/{best_model_name.replace(" ", "_").lower()}_model.joblib')
    
    import joblib
    joblib.dump(data['scaler'], 'models/scaler.joblib')
    joblib.dump(data['encoders'], 'models/encoders.joblib')
    joblib.dump(data['bmi_median'], 'models/bmi_median.joblib')
    joblib.dump(data['feature_names'], 'models/feature_names.joblib')
    
    patient_data_processed = sample_patient()
    
    prediction = predict_stroke(best_model, patient_data_processed, scaler = data['scaler'])
    
    print(f"Stroke Prediction: {'YES' if prediction['stroke_prediction'] == 1 else 'NO'}")
    if prediction['stroke_probability'] is not None:
        print(f"Stroke Probability: {prediction['stroke_probability']:.2%}")
    print(f"Risk Level: {prediction['risk_level']}")
    patient_data_raw = sample_patient_raw()
    prediction_raw = predict_stroke(
        best_model, 
        patient_data_raw, 
        scaler = data['scaler'], 
        encoders = data['encoders'], 
        bmi_median = data['bmi_median']
    )

    print(f"Stroke Prediction: {'YES' if prediction_raw['stroke_prediction'] == 1 else 'NO'}")
    if prediction_raw['stroke_probability'] is not None:
        print(f"Stroke Probability: {prediction_raw['stroke_probability']:.2%}")
    print(f"Risk Level: {prediction_raw['risk_level']}")
    print(f"\nRecommendation:\n{prediction_raw['recommendation']}")
    


if __name__ == "__main__":
    main()