#!/usr/bin/env python3
"""
preprocess.py
Module de pretraitement des donnees medicales.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data():
    """
    Charge le dataset UCI Heart Disease depuis Internet.
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
               'restecg', 'thalach', 'exang', 'oldpeak',
               'slope', 'ca', 'thal', 'target']
    df = pd.read_csv(url, names=columns)
    return df


def clean_data(df):
    """
    Nettoie les donnees.
    """
    df = df.replace('?', np.nan)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    df['target'] = (df['target'] > 0).astype(int)
    return df


def split_and_scale(df):
    """
    Separe les donnees en train/test et standardise.
    """
    X = df.drop('target', axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler, X_train_scaled, X_test_scaled


def get_feature_names():
    return ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak',
            'slope', 'ca', 'thal']