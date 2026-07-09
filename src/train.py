#!/usr/bin/env python3
"""
train.py
Module d'entraînement et d'évaluation des modèles.
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score)
import pickle
import os


def get_models():
    """
    Retourne les 4 modèles de classification.
    """
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
    }


def train_and_evaluate(models, X_train, X_test, y_train, y_test, 
                       X_train_scaled, X_test_scaled):
    """
    Entraîne chaque modèle et calcule les métriques.
    """
    results = {}
    
    for name, model in models.items():
        print(f"\n🤖 Entraînement : {name}")
        
        # SVM et Logistic Regression ont besoin de données standardisées
        if name in ['SVM', 'Logistic Regression']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calcul des 5 métriques
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1-Score': f1_score(y_test, y_pred),
            'ROC-AUC': roc_auc_score(y_test, y_proba)
        }
        
        print(f"   Accuracy: {results[name]['Accuracy']:.4f}")
        print(f"   F1-Score: {results[name]['F1-Score']:.4f}")
    
    # Meilleur modèle = celui avec le meilleur F1-Score
    best_name = max(results, key=lambda x: results[x]['F1-Score'])
    
    print(f"\n🏆 MEILLEUR MODÈLE : {best_name}")
    
    return results, best_name


def save_best_model(model, model_name, filepath='models/best_model.pkl'):
    """Sauvegarde le modèle avec pickle."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump({'model': model, 'name': model_name}, f)
    print(f"💾 Modèle sauvegardé : {filepath}")


def print_results_table(results):
    """Affiche un tableau comparatif."""
    df = pd.DataFrame(results).T
    print("\n" + "=" * 60)
    print("📊 TABLEAU COMPARATIF")
    print("=" * 60)
    print(df.round(4).to_string())