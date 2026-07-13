#!/usr/bin/env python3
"""
main.py - VERSION CORRIGÉE
Sauvegarde le modèle ET le scaler ensemble
"""

import sys
import os
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import load_data, clean_data, split_and_scale, get_feature_names
from train import get_models, train_and_evaluate, print_results_table
from evaluate import (plot_eda, plot_model_comparison, plot_confusion_matrix,
                      plot_roc_curves, plot_feature_importance)
from predict import predict_disease, EXAMPLE_PATIENTS

def main():
    print("=" * 60)
    print("🩺 CODEALPHA — PRÉDICTION DE MALADIES CARDIAQUES")
    print("=" * 60)
    
    # ── ÉTAPE 1 : Chargement ──
    print("\n📦 ÉTAPE 1 : Chargement et nettoyage")
    df_raw = load_data()
    df = clean_data(df_raw)
    print(f"   Données : {df.shape[0]} patients")
    
    X_train, X_test, y_train, y_test, scaler, X_train_s, X_test_s = split_and_scale(df)
    print(f"   📚 Train : {X_train.shape[0]} | 🧪 Test : {X_test.shape[0]}")
    
    # ── ÉTAPE 2 : EDA ──
    print("\n📊 ÉTAPE 2 : Visualisations EDA")
    plot_eda(df)
    
    # ── ÉTAPE 3 : Entraînement ──
    print("\n🤖 ÉTAPE 3 : Entraînement des modèles")
    models = get_models()
    results, best_name = train_and_evaluate(
        models, X_train, X_test, y_train, y_test, X_train_s, X_test_s
    )
    print_results_table(results)
    
    # ── ÉTAPE 4 : Évaluation ──
    print("\n📈 ÉTAPE 4 : Visualisations")
    plot_model_comparison(results)
    
    best_model = models[best_name]
    if best_name in ['SVM', 'Logistic Regression']:
        y_pred = best_model.predict(X_test_s)
    else:
        y_pred = best_model.predict(X_test)
    
    plot_confusion_matrix(y_test, y_pred, best_name)
    plot_roc_curves(models, X_test, X_test_s, y_test)
    plot_feature_importance(X_train, y_train, get_feature_names())
    
    # ── ÉTAPE 5 : Sauvegarde du modèle ET du scaler ENSEMBLE ──
    print("\n💾 ÉTAPE 5 : Sauvegarde du modèle et du scaler")
    
    # Sauvegarder modèle + scaler dans UN SEUL fichier
    model_data = {
        'model': best_model,
        'name': best_name,
        'scaler': scaler  # 🔥 SCALER INCLUS !
    }
    
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_model.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"   ✅ Modèle + scaler sauvegardés : {model_path}")
    
    # Sauvegarder aussi le scaler séparément (pour app.py)
    scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"   ✅ Scaler séparé : {scaler_path}")
    
    # ── ÉTAPE 6 : Test de prédiction ──
    print("\n🔮 ÉTAPE 6 : Test avec patient à risque")
    
    # Charger et tester
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    test_model = data['model']
    test_scaler = data['scaler']
    
    # Patient à risque élevé
    patient_risque = pd.DataFrame([{
        'age': 65, 'sex': 1, 'cp': 4, 'trestbps': 150,
        'chol': 280, 'fbs': 1, 'restecg': 2, 'thalach': 120,
        'exang': 1, 'oldpeak': 3.5, 'slope': 2, 'ca': 2, 'thal': 7
    }])
    
    patient_scaled = test_scaler.transform(patient_risque)
    proba = test_model.predict_proba(patient_scaled)[0][1]
    print(f"   Patient à risque : {proba*100:.1f}% (devrait être > 70%)")
    
    # Patient sain
    patient_sain = pd.DataFrame([{
        'age': 40, 'sex': 0, 'cp': 1, 'trestbps': 120,
        'chol': 200, 'fbs': 0, 'restecg': 0, 'thalach': 170,
        'exang': 0, 'oldpeak': 0.0, 'slope': 1, 'ca': 0, 'thal': 3
    }])
    
    patient_scaled = test_scaler.transform(patient_sain)
    proba = test_model.predict_proba(patient_scaled)[0][1]
    print(f"   Patient sain : {proba*100:.1f}% (devrait être < 30%)")
    
    print("\n" + "=" * 60)
    print("✅ PIPELINE TERMINÉ !")
    print(f"🏆 Meilleur modèle : {best_name}")
    print("=" * 60)

if __name__ == "__main__":
    import pandas as pd
    main()