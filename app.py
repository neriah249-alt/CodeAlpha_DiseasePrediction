#!/usr/bin/env python3
"""
app.py - AVEC VALIDATION STRICTE
"""

import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Prediction Cardiaque", page_icon="🩺")

st.title("🩺 Prediction de Maladies Cardiaques")
st.write("Application ML — CodeAlpha Internship")

# Charger le modele
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_model.pkl')
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['name']

try:
    model, model_name = load_model()
    st.sidebar.success(f"Modele charge: {model_name}")
    model_ok = True
except Exception as e:
    st.sidebar.error(f"Erreur modele: {str(e)}")
    model_ok = False

# Formulaire
with st.form("form"):
    st.subheader("Informations du patient")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=55)
        sex = st.selectbox("Sexe", [0, 1], format_func=lambda x: "Femme" if x==0 else "Homme")
        cp = st.selectbox("Douleur thoracique", [1,2,3,4])
        trestbps = st.number_input("Tension (mmHg)", min_value=1, max_value=300, value=130)
    with col2:
        chol = st.number_input("Cholesterol (mg/dl)", min_value=1, max_value=1000, value=240)
        fbs = st.selectbox("Glycemie > 120", [0,1], format_func=lambda x: "Non" if x==0 else "Oui")
        restecg = st.selectbox("ECG", [0,1,2])
        thalach = st.number_input("FC max", min_value=1, max_value=300, value=150)
    with col3:
        exang = st.selectbox("Angine", [0,1], format_func=lambda x: "Non" if x==0 else "Oui")
        oldpeak = st.number_input("Depression ST", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
        slope = st.selectbox("Pente ST", [1,2,3])
        ca = st.number_input("Vaisseaux", min_value=0, max_value=10, value=0)
        thal = st.selectbox("Thalassemie", [3,6,7])
    
    submit = st.form_submit_button("Predire", type="primary")

# VALIDATION MANUELLE APRES SOUMISSION
if submit:
    errors = []
    
    # Verifier chaque valeur
    if age < 20 or age > 100:
        errors.append(f"❌ Age: {age} — Doit etre entre 20 et 100 ans")
    if trestbps < 90 or trestbps > 200:
        errors.append(f"❌ Tension: {trestbps} — Doit etre entre 90 et 200 mmHg")
    if chol < 100 or chol > 600:
        errors.append(f"❌ Cholesterol: {chol} — Doit etre entre 100 et 600 mg/dl")
    if thalach < 70 or thalach > 220:
        errors.append(f"❌ FC max: {thalach} — Doit etre entre 70 et 220 bpm")
    if ca < 0 or ca > 3:
        errors.append(f"❌ Vaisseaux: {ca} — Doit etre entre 0 et 3")
    if oldpeak < 0 or oldpeak > 6:
        errors.append(f"❌ Depression ST: {oldpeak} — Doit etre entre 0.0 et 6.0")
    if thal not in [3, 6, 7]:
        errors.append(f"❌ Thalassemie: {thal} — Doit etre 3, 6 ou 7")
    
    # Afficher les erreurs
    if errors:
        st.error("**Erreurs de validation detectees :**")
        for error in errors:
            st.write(error)
        st.info("Veuillez corriger les valeurs incorrectes et reessayer.")
    
    elif not model_ok:
        st.error("Le modele n'est pas charge.")
    
    else:
        # Tout est valide, on predit
        df = pd.DataFrame([{
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
            'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
            'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
        }])
        
        proba = model.predict_proba(df)[0][1]
        pred = "Maladie" if proba > 0.5 else "Sain"
        
        st.markdown("---")
        if proba > 0.7:
            st.error(f"🔴 RISQUE ELEVE: {proba*100:.1f}%")
        elif proba > 0.4:
            st.warning(f"🟡 RISQUE MODERE: {proba*100:.1f}%")
        else:
            st.success(f"🟢 RISQUE FAIBLE: {proba*100:.1f}%")
        
        st.metric("Prediction", pred)
        st.progress(float(proba))