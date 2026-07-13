#!/usr/bin/env python3
"""
app.py - AVEC SLIDERS (bloque les valeurs impossibles)
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

# Avertissement medical
st.info("⚠️ Cette application est a titre demonstratif. Consultez toujours un medecin.")

# Formulaire avec SLIDERS (impossible de depasser les limites)
with st.form("form"):
    st.subheader("Informations du patient")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # SLIDER : impossible de mettre 50000
        age = st.slider("Age (ans)", min_value=20, max_value=100, value=55, step=1)
        sex = st.selectbox("Sexe", [0, 1], format_func=lambda x: "Femme" if x==0 else "Homme")
        cp = st.selectbox("Douleur thoracique", [1,2,3,4],
                         format_func=lambda x: {1:"Typique",2:"Atypique",3:"Non-angineuse",4:"Asymptomatique"}[x])
        # SLIDER pour tension
        trestbps = st.slider("Tension (mmHg)", min_value=90, max_value=200, value=130, step=1)
    
    with col2:
        # SLIDER pour cholesterol
        chol = st.slider("Cholesterol (mg/dl)", min_value=100, max_value=600, value=240, step=1)
        fbs = st.selectbox("Glycemie > 120", [0,1], format_func=lambda x: "Non" if x==0 else "Oui")
        restecg = st.selectbox("ECG", [0,1,2],
                              format_func=lambda x: {0:"Normal",1:"Anomalie ST-T",2:"Hypertrophie"}[x])
        # SLIDER pour FC max
        thalach = st.slider("FC max (bpm)", min_value=70, max_value=220, value=150, step=1)
    
    with col3:
        exang = st.selectbox("Angine induite", [0,1], format_func=lambda x: "Non" if x==0 else "Oui")
        # SLIDER pour depression ST
        oldpeak = st.slider("Depression ST", min_value=0.0, max_value=6.0, value=1.5, step=0.1)
        slope = st.selectbox("Pente ST", [1,2,3],
                            format_func=lambda x: {1:"Montante",2:"Plate",3:"Descendante"}[x])
        # SLIDER pour vaisseaux
        ca = st.slider("Vaisseaux colores", min_value=0, max_value=3, value=0, step=1)
        thal = st.selectbox("Thalassemie", [3,6,7],
                           format_func=lambda x: {3:"Normal",6:"Defaut fixe",7:"Defaut reversible"}[x])
    
    submit = st.form_submit_button("Predire", type="primary")

# Prediction
if submit and model_ok:
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
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Probabilite", f"{proba*100:.1f}%")
    with col2:
        st.metric("Prediction", pred)
    with col3:
        st.metric("Modele", model_name)
    
    st.progress(float(proba))