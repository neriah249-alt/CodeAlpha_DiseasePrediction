#!/usr/bin/env python3
"""
predict.py
Module de prédiction pour de nouveaux patients.
"""

import pickle
import pandas as pd


def load_model(filepath='models/best_model.pkl'):
    """Charge le modèle sauvegardé."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def predict_disease(patient_data, model_dict, scaler):
    """
    Prédit la probabilité de maladie pour un patient.
    
    patient_data = dict avec les 13 features
    """
    model = model_dict['model']
    model_name = model_dict['name']
    
    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                     'restecg', 'thalach', 'exang', 'oldpeak', 
                     'slope', 'ca', 'thal']
    
    df_patient = pd.DataFrame([patient_data], columns=feature_names)
    
    # Standardiser si nécessaire
    if model_name in ['SVM', 'Logistic Regression']:
        df_patient = scaler.transform(df_patient)
        proba = model.predict_proba(df_patient)[0][1]
    else:
        proba = model.predict_proba(df_patient)[0][1]
    
    prediction = 1 if proba > 0.5 else 0
    
    return {
        'prediction': 'Maladie détectée' if prediction == 1 else 'Pas de maladie',
        'probabilité': round(proba * 100, 2),
        'model_utilisé': model_name
    }


# Exemples de patients pour tester
EXAMPLE_PATIENTS = {
    'patient_sain': {
        'age': 40, 'sex': 0, 'cp': 1, 'trestbps': 120,
        'chol': 200, 'fbs': 0, 'restecg': 0, 'thalach': 170,
        'exang': 0, 'oldpeak': 0.0, 'slope': 1, 'ca': 0, 'thal': 3
    },
    'patient_risque': {
        'age': 65, 'sex': 1, 'cp': 4, 'trestbps': 150,
        'chol': 280, 'fbs': 1, 'restecg': 2, 'thalach': 120,
        'exang': 1, 'oldpeak': 3.5, 'slope': 2, 'ca': 2, 'thal': 7
    }
}