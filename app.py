#!/usr/bin/env python3
"""
app.py - VERSION CORRIGÉE AVEC SCALER INTÉGRÉ
"""

import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Prediction Cardiaque", page_icon="🩺")

st.title("🩺 Prediction de Maladies Cardiaques")
st.write("Application ML — CodeAlpha Internship")

# Charger le modèle ET le scaler (même fichier)
@st.cache_resource
def load_model():
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, 'models', 'best_model.pkl')
    
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    
    return data['model'], data['name'], data['scaler']

try:
    model, model_name, scaler = load_model()
    st.sidebar.success(f"✅ Modèle: {model_name}")
    model_ok = True
except Exception as e:
    st.sidebar.error(f"❌ Erreur: {str(e)}")
    model_ok = False

# Formulaire
with st.form("form"):
    st.subheader("Informations du patient")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Âge (ans)", 20, 100, 55)
        sex = st.selectbox("Sexe", [0, 1], format_func=lambda x: "Femme" if x==0 else "Homme")
        cp = st.selectbox("Douleur thoracique", [1,2,3,4],
                         format_func=lambda x: {1:"Typique",2:"Atypique",3:"Non-angineuse",4:"Asymptomatique"}[x])
        trestbps = st.slider("Tension (mmHg)", 90, 200, 130)
    with col2:
        chol = st.slider("Cholestérol (mg/dl)", 100, 600, 240)
        fbs = st.selectbox("Glycémie > 120", [0,1], format_func=lambda x: "Non" if x==0 else "Oui")
        restecg = st.selectbox("ECG", [0,1,2],
                              format_func=lambda x: {0:"Normal",1:"Anomalie ST-T",2:"Hypertrophie"}[x])
        thalach = st.slider("FC max (bpm)", 70, 220, 150)
    with col3:
        exang = st.selectbox("Angine induite", [0,1], format_func=lambda x: "Non" if x==0 else "Oui")
        oldpeak = st.slider("Dépression ST", 0.0, 6.0, 1.5, 0.1)
        slope = st.selectbox("Pente ST", [1,2,3],
                            format_func=lambda x: {1:"Montante",2:"Plate",3:"Descendante"}[x])
        ca = st.slider("Vaisseaux colorés", 0, 3, 0)
        thal = st.selectbox("Thalassémie", [3,6,7],
                           format_func=lambda x: {3:"Normal",6:"Défaut fixe",7:"Défaut réversible"}[x])
    
    submit = st.form_submit_button("Prédire", type="primary")

# Prediction
if submit and model_ok:
    df = pd.DataFrame([{
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }])
    
    # STANDARDISER avec le scaler du modèle
    df_scaled = scaler.transform(df)
    
    # Prédire
    proba = model.predict_proba(df_scaled)[0][1]
    pred = "Maladie" if proba > 0.5 else "Sain"
    
    # Affichage
    st.markdown("---")
    if proba > 0.7:
        st.error(f"🔴 RISQUE ÉLEVÉ: {proba*100:.1f}%")
    elif proba > 0.4:
        st.warning(f"🟡 RISQUE MODÉRÉ: {proba*100:.1f}%")
    else:
        st.success(f"🟢 RISQUE FAIBLE: {proba*100:.1f}%")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Probabilité", f"{proba*100:.1f}%")
    with col2:
        st.metric("Prédiction", pred)
    with col3:
        st.metric("Modèle", model_name)
    
    st.progress(float(proba))

st.markdown("---")
st.caption("🎓 Projet CodeAlpha — Machine Learning Internship")