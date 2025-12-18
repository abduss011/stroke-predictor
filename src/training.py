import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve)
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

def handling(X_train,y_train):
    smote = SMOTE(random_state = 42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def mx(X_train,y_train):
    model = LogisticRegression(random_state = 42, max_iter = 1000, class_weight = 'balanced')
    model.fit(X_train, y_train)
    return model


def forest(X_train,y_train):
    model = RandomForestClassifier(n_estimators = 100,max_depth = 10, random_state = 42,class_weight = 'balanced')
    model.fit(X_train, y_train)
    return model


def dec_tree(X_train,y_train):
    model = DecisionTreeClassifier(max_depth = 10,random_state = 42,class_weight = 'balanced')
    model.fit(X_train, y_train)
    return model


def grd(X_train,y_train):
    model = GradientBoostingClassifier(n_estimators = 100,learning_rate = 0.1,max_depth = 5,random_state = 42)
    model.fit(X_train, y_train)
    return model


def xgboost_model(X_train, y_train):
    if not XGB_AVAILABLE:
        return None
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(n_estimators = 100,learning_rate = 0.1, max_depth = 5,scale_pos_weight = scale_pos_weight,random_state = 42,eval_metric = 'logloss')
    model.fit(X_train, y_train)
    return model


def svm(X_train, y_train):
    model = SVC(kernel = 'rbf',probability = True,random_state = 42,class_weight = 'balanced')
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name = 'Model'):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    else:
        roc_auc = None
    
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n--- {model_name} ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    return metrics

def train_models(X_train, y_train, X_test, y_test, use_smote = True):
    if use_smote:
        X_train_resampled, y_train_resampled = handling(X_train, y_train)
    else:
        X_train_resampled, y_train_resampled = X_train, y_train

    models = {}
    all_metrics = []
    
    models['Logistic Regression'] = mx(X_train_resampled, y_train_resampled)
    all_metrics.append(evaluate_model(models['Logistic Regression'], X_test, y_test, 'Logistic Regression'))
    
    models['Decision Tree'] = dec_tree(X_train_resampled, y_train_resampled)
    all_metrics.append(evaluate_model(models['Decision Tree'], X_test, y_test, 'Decision Tree'))
    
    models['Random Forest'] = forest(X_train_resampled, y_train_resampled)
    all_metrics.append(evaluate_model(models['Random Forest'], X_test, y_test, 'Random Forest'))
    
    models['Gradient Boosting'] = grd(X_train_resampled, y_train_resampled)
    all_metrics.append(evaluate_model(models['Gradient Boosting'], X_test, y_test, 'Gradient Boosting'))
    
    if XGB_AVAILABLE:
        models['XGBoost'] = xgboost_model(X_train_resampled, y_train_resampled)
        all_metrics.append(evaluate_model(models['XGBoost'], X_test, y_test, 'XGBoost'))
    else:
        print("\nSkipping XGBoost (module not installed)")

    models['SVM'] = svm(X_train_resampled, y_train_resampled)
    all_metrics.append(evaluate_model(models['SVM'], X_test, y_test, 'SVM'))

    comparison_df = pd.DataFrame([
        {
            'Model': m['model_name'],
            'Accuracy': f"{m['accuracy']:.4f}",
            'Precision': f"{m['precision']:.4f}",
            'Recall': f"{m['recall']:.4f}",
            'F1-Score': f"{m['f1_score']:.4f}",
            'ROC-AUC': f"{m['roc_auc']:.4f}" if m['roc_auc'] else 'N/A'
        }
        for m in all_metrics
    ])
    
    print("\nModel Comparison Table:")
    print(comparison_df.to_string(index = False))
    
    best_model_idx = np.argmax([m['f1_score'] for m in all_metrics])
    best_model_name = all_metrics[best_model_idx]['model_name']
    
    print(f"\nBest Model by F1-Score: {best_model_name} ({all_metrics[best_model_idx]['f1_score']:.4f})")

    return {
        'models': models,
        'metrics': all_metrics,
        'best_model': best_model_name,
        'comparison': comparison_df
    }


def save_model(model,filepath = 'models/stroke_model.joblib'):
    """Saves the fitted model to a file."""
    joblib.dump(model,filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath = 'models/stroke_model.joblib'):
    """Loads a model from a file."""
    model = joblib.load(filepath)
    print(f"Model loaded from {filepath}")
    return model