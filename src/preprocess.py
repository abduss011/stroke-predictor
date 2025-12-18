import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath = 'data/healthcare-dataset-stroke-data.csv'):
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

def miss(df, bmi_median=None):
    df = df.copy()
    if 'bmi' in df.columns:
        if bmi_median is None:
            bmi_median = df['bmi'].median()
        df['bmi'].fillna(bmi_median, inplace = True)
    df.dropna(inplace = True)
    return df, bmi_median

def encode(df):
    df = df.copy()
    encoders = {}
    categories = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    for col in categories:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders

def create_features(df):
    df = df.copy()
    df['age_group'] = pd.cut(df['age'], bins = [0, 18, 35, 50, 65, 100], labels = [0, 1, 2, 3, 4])
    df['age_group'] = df['age_group'].astype(int)
    if 'bmi' in df.columns:
        df['bmi_category'] = pd.cut(df['bmi'], bins = [0, 18.5, 25, 30, 100], labels = [0, 1, 2, 3])
        df['bmi_category'] = df['bmi_category'].astype(int)
    if 'avg_glucose_level' in df.columns:
        df['high_glucose'] = (df['avg_glucose_level'] > 125).astype(int)
    if 'hypertension' in df.columns and 'heart_disease' in df.columns:
        df['cardio_risk'] = df['hypertension'] + df['heart_disease']
    return df


def prepare_features(df, target_column = 'stroke'):
    df = df.copy()
    
    columns_to_drop = ['id'] if 'id' in df.columns else []
    if target_column in df.columns:
        columns_to_drop.append(target_column)
    
    X = df.drop(columns = columns_to_drop, errors = 'ignore')
    y = df[target_column] if target_column in df.columns else None
    
    feature_names = X.columns.tolist()

    if y is not None:
        print(f"\nTarget distribution:")
        print(y.value_counts())
    
    return X, y, feature_names


def scalee(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features scaled successfully")
    
    return X_train_scaled, X_test_scaled, scaler


def split_data(X, y, test_size = 0.2, random_state = 42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state, stratify = y)
    print(f"Data split: {len(X_train)}, {len(X_test)} samples")

    return X_train, X_test, y_train, y_test


def preprocessing(filepath = 'data/healthcare-dataset-stroke-data.csv', test_size = 0.2):
    df = load_data(filepath)
    if df is None:
        return None
    
    df, bmi_median = miss(df)
    
    df, encoders = encode(df)
    
    df = create_features(df)
    
    X, y, feature_names = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size = test_size)
    X_train_scaled, X_test_scaled, scaler = scalee(X_train, X_test)
    
    return {
        'X_train': X_train_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': feature_names,
        'scaler': scaler,
        'encoders': encoders,
        'bmi_median': bmi_median
    }