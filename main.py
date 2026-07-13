#!/usr/bin/env python3
"""
main.py
Script principal — exécute tout le pipeline ML + sauvegarde le scaler
Usage: python main.py
"""

import sys
import os
import pickle

# Ajouter src/ au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import load_data, clean_data, split_and_scale, get_feature_names
from train import get_models, train_and_evaluate, save_best_model, print_results_table
from evaluate import (plot_eda, plot_model_comparison, plot_confusion_matrix,
                      plot_roc_curves, plot_feature_importance)
from predict import predict_disease, EXAMPLE_PATIENTS, load_model


def main():
    print("=" * 60)
    print("🩺 CODEALPHA — PRÉDICTION DE MALADIES CARDIAQUES")
    print("=" * 60)
    
    # ── ÉTAPE 1 : Chargement ──
    print("\n📦 ÉTAPE 1 : Chargement et nettoyage")
    df_raw = load_data()
    df = clean_data(df_raw)
    print(f"   Données : {df.shape[0]} patients, {df.shape[1]-1} features")
    print(f"   Sain : {df['target'].value_counts()[0]} | Malade : {df['target'].value_counts()[1]}")
    
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
    print("\n📈 ÉTAPE 4 : Visualisations d'évaluation")
    plot_model_comparison(results)
    
    best_model = models[best_name]
    if best_name in ['SVM', 'Logistic Regression']:
        y_pred = best_model.predict(X_test_s)
    else:
        y_pred = best_model.predict(X_test)
    
    plot_confusion_matrix(y_test, y_pred, best_name)
    plot_roc_curves(models, X_test, X_test_s, y_test)
    plot_feature_importance(X_train, y_train, get_feature_names())
    
    # ── ÉTAPE 5 : Sauvegarde du modèle ET du scaler ──
    print("\n💾 ÉTAPE 5 : Sauvegarde du modèle et du scaler")
    save_best_model(best_model, best_name)
    
    # 🔥 SAUVEGARDER LE SCALER (nouveau !)
    scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'scaler.pkl')
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"   ✅ Scaler sauvegardé : {scaler_path}")
    
    # ── ÉTAPE 6 : Prédictions ──
    print("\n🔮 ÉTAPE 6 : Prédictions sur exemples")
    model_dict = load_model()
    
    for name, data in EXAMPLE_PATIENTS.items():
        result = predict_disease(data, model_dict, scaler)
        print(f"\n👤 {name}:")
        print(f"   → 🩺 {result['prediction']} ({result['probabilité']}%)")
    
    print("\n" + "=" * 60)
    print("✅ PIPELINE TERMINÉ !")
    print(f"🏆 Meilleur modèle : {best_name}")
    print("=" * 60)


if __name__ == "__main__":
    main()