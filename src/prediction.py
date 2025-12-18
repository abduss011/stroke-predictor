import numpy as np
import pandas as pd
import joblib
from src.preprocess import create_features

def process_input(patient_data, encoders, bmi_median, scaler = None, feature_names = None):
    if isinstance(patient_data, dict):
        df = pd.DataFrame([patient_data])
    else:
        df = patient_data.copy()

    if 'bmi' in df.columns:
        df['bmi'] = df['bmi'].fillna(bmi_median)

    for col, le in encoders.items():
        if col in df.columns:
            try:
                df[col] = df[col].apply(lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1)
            except Exception as e:
                print(f"Warning: Issue encoding column {col}: {e}")
                
    df = create_features(df)
    
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
        
    if scaler is not None:
        if feature_names is not None:   
            df = df[feature_names]
        
        features = scaler.transform(df)
    else:
        features = df.values
        
    return features


def predict_stroke(model, patient_data, encoders=None, bmi_median=None, scaler=None, feature_names=None, threshold=0.5):
    if encoders is not None and bmi_median is not None:
        patient_features = process_input(patient_data, encoders, bmi_median, scaler, feature_names)
    else:
        if isinstance(patient_data, dict):
            patient_df = pd.DataFrame([patient_data])
        else:
            patient_df = patient_data.copy()
            
        if scaler is not None:
            patient_features = scaler.transform(patient_df)
        else:
            patient_features = patient_df.values
    
    pred_class = model.predict(patient_features)[0]
    
    if hasattr(model, 'predict_proba'):
        pred_proba = model.predict_proba(patient_features)[0]
        stroke_probability = pred_proba[1]
    else:
        stroke_probability = None
    
    if stroke_probability is not None:
        if stroke_probability < 0.3:
            risk_level = "Low"
        elif stroke_probability < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"
    else:
        risk_level = "High" if pred_class == 1 else "Low"

    return {
        'stroke_prediction': int(pred_class),
        'stroke_probability': float(stroke_probability) if stroke_probability is not None else None,
        'risk_level': risk_level,
        'recommendation': get_recommendation(pred_class, stroke_probability)
    }



def predict_prob(model, patient_data, scaler = None):
    if isinstance(patient_data, dict):
        patient_df = pd.DataFrame([patient_data])
    else:
        patient_df = patient_data.copy()

    if scaler is not None:
        patient_features = scaler.transform(patient_df)
    else:
        patient_features = patient_df.values
    
    if hasattr(model, 'predict_proba'):
        pred_proba = model.predict_proba(patient_features)[0]
        return pred_proba[1]
    else:
        return None
    
def get_recommendation(prediction, probability = None):
    if prediction == 1 or (probability is not None and probability > 0.6):
        return ("HIGH RISK: Immediate medical consultation recommended. "
                "Monitor blood pressure, glucose levels, and lifestyle factors closely.")
    elif probability is not None and probability > 0.3:
        return ("MODERATE RISK: Regular health check-ups recommended. "
                "Maintain healthy lifestyle, exercise regularly, and monitor vital signs.")
    else:
        return ("LOW RISK: Continue maintaining healthy lifestyle. "
                "Regular exercise, balanced diet, and annual health check-ups recommended.")

def batch_predict(model, data, scaler = None):
    if scaler is not None:
        features = scaler.transform(data)
    else:
        features = data.values
    predictions = model.predict(features)
    
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features)[:, 1]
    else:
        probabilities = None
    
    results = pd.DataFrame({
        'stroke_prediction': predictions
    })
    
    if probabilities is not None:
        results['stroke_probability'] = probabilities
        results['risk_level'] = pd.cut(probabilities, bins = [0, 0.3, 0.6, 1.0], labels = ['Low', 'Medium', 'High'])
    
    return results

def sample_patient():
    patient = {
        'gender': 1,
        'age': 67,
        'hypertension': 0,
        'heart_disease': 1,
        'ever_married': 1,
        'work_type': 2,
        'Residence_type': 1,
        'avg_glucose_level': 228.69,
        'bmi': 36.6,
        'smoking_status': 1,
        'age_group': 3,
        'bmi_category': 3,
        'high_glucose': 1,
        'cardio_risk': 1
    }
    return patient

def sample_patient_raw():
    patient = {
        'gender': 'Male',
        'age': 67,
        'hypertension': 0,
        'heart_disease': 1,
        'ever_married': 'Yes',
        'work_type': 'Private',
        'Residence_type': 'Urban',
        'avg_glucose_level': 228.69,
        'bmi': 36.6,
        'smoking_status': 'formerly smoked'
    }
    return patient