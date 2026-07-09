#!/usr/bin/env python3
"""
evaluate.py
Module de visualisation et d'évaluation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import os


def plot_eda(df, save_path='results/eda_heart_disease.png'):
    """Graphiques d'analyse exploratoire."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Distribution de l'âge
    sns.histplot(data=df, x='age', hue='target', kde=True, ax=axes[0,0])
    axes[0,0].set_title("Distribution de l'âge par maladie")
    
    # 2. Matrice de corrélation
    sns.heatmap(df.corr(), annot=True, cmap='RdYlBu_r', fmt='.2f', ax=axes[0,1])
    axes[0,1].set_title('Matrice de corrélation')
    
    # 3. Cholestérol vs Âge
    sns.scatterplot(data=df, x='age', y='chol', hue='target', ax=axes[1,0])
    axes[1,0].set_title('Cholestérol vs Âge')
    
    # 4. Fréquence cardiaque max
    sns.boxplot(data=df, x='target', y='thalach', ax=axes[1,1])
    axes[1,1].set_title('Fréquence cardiaque max')
    axes[1,1].set_xticklabels(['Sain', 'Malade'])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 EDA sauvegardé : {save_path}")


def plot_model_comparison(results, save_path='results/model_comparison.png'):
    """Bar chart comparatif des modèles."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    df = pd.DataFrame(results).T
    
    fig, ax = plt.subplots(figsize=(12, 6))
    df.plot(kind='bar', ax=ax)
    plt.title('Comparaison des modèles')
    plt.ylabel('Score')
    plt.xticks(rotation=30)
    plt.legend(loc='lower right')
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Comparaison sauvegardée : {save_path}")


def plot_confusion_matrix(y_true, y_pred, model_name, save_path='results/confusion_matrix.png'):
    """Matrice de confusion."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Sain', 'Malade'],
                yticklabels=['Sain', 'Malade'])
    plt.title(f'Matrice de Confusion - {model_name}')
    plt.xlabel('Prédiction')
    plt.ylabel('Réalité')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Matrice de confusion sauvegardée : {save_path}")


def plot_roc_curves(models, X_test, X_test_scaled, y_test, save_path='results/roc_curves.png'):
    """Courbes ROC de tous les modèles."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for (name, model), color in zip(models.items(), colors):
        if name in ['SVM', 'Logistic Regression']:
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_proba = model.predict_proba(X_test)[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, color=color, linewidth=2, label=f'{name} (AUC={auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Aléatoire (AUC=0.500)')
    plt.xlabel('Faux Positifs')
    plt.ylabel('Vrais Positifs')
    plt.title('Courbes ROC')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Courbes ROC sauvegardées : {save_path}")


def plot_feature_importance(X_train, y_train, feature_names, save_path='results/feature_importance.png'):
    """Importance des features avec Random Forest."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance, x='Importance', y='Feature')
    plt.title('Importance des features')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 Importance des features sauvegardée : {save_path}")
    
    return importance