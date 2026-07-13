#!/usr/bin/env python3
"""
app.py - AVEC STANDARDISATION CORRECTE
"""

import streamlit as st
import pandas as pd
import pickle
import os

st.set_page_config(page_title="Prediction Cardiaque", page_icon="🩺")

st.title("🩺 Prediction de Maladies Cardiaques")
st.write("Application ML — CodeAlpha Internship")

# Charger le modele ET le scaler
@st.cache_resource
def load_model_and_scaler():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_model.pkl')
    
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)
    
    # Charger le scaler sauvegarde separement
    scaler_path = os.path.join(os.path.dirname(__file__), 'models', 'scaler.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    else:
        # Si pas de scaler sauvegarde, le recreer
        from preprocess import load_data, clean_data, split_and_scale
        df = clean_data(load_data())
        _, _, _, _, scaler, _, _ = split_and_scale(df)
        # Sauvegarder pour la prochaine fois
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
    
    return model_dict, scaler

try:
    model_dict, scaler = load_model_and_scaler()
    model = model_dict['model']
    model_name = model_dict['name']
    st.sidebar.success(f"Modele: {model_name}")
    model_ok = True
except Exception as e:
    st.sidebar.error(f"Erreur: {str(e)}")
    model_ok = False

# Avertissement
st.info("⚠️ Application demonstrative. Consultez un medecin.")

# Formulaire
with st.form("form"):
    st.subheader("Informations du patient")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.slider("Age (ans)", 20, 100, 55)
        sex = st.selectbox("Sexe", [0, 1], format_func=lambda x: "Femme" if x==0 else "Homme")
        cp = st.selectbox("Douleur thoracique", [1,2,3,4],
                         format_func=lambda x: {1:"Typique",2:"Atypique",3:"Non-angineuse",4:"Asymptomatique"}[x])
        trestbps = st.slider("Tension (mmHg)", 90, 200, 130)
    
    with col2:
        chol = st.slider("Cholesterol (mg/dl)", 100, 600, 240)
        fbs = st.selectbox("Glycemie > 120", [0,1], format_func=lambda x: "Non" if x==0 else "Oui")
        restecg = st.selectbox("ECG", [0,1,2],
                              format_func=lambda x: {0:"Normal",1:"Anomalie ST-T",2:"Hypertrophie"}[x])
        thalach = st.slider("FC max (bpm)", 70, 220, 150)
    
    with col3:
        exang = st.selectbox("Angine induite", [0,1], format_func=lambda x: "Non" if x==0 else "Oui")
        oldpeak = st.slider("Depression ST", 0.0, 6.0, 1.5, 0.1)
        slope = st.selectbox("Pente ST", [1,2,3],
                            format_func=lambda x: {1:"Montante",2:"Plate",3:"Descendante"}[x])
        ca = st.slider("Vaisseaux colores", 0, 3, 0)
        thal = st.selectbox("Thalassemie", [3,6,7],
                           format_func=lambda x: {3:"Normal",6:"Defaut fixe",7:"Defaut reversible"}[x])
    
    submit = st.form_submit_button("Predire", type="primary")

# Prediction AVEC STANDARDISATION
if submit and model_ok:
    # Creer DataFrame
    df = pd.DataFrame([{
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }])
    
    # STANDARDISER les donnees (obligatoire pour SVM)
    df_scaled = scaler.transform(df)
    
    # Predire avec les donnees standardisees
    proba = model.predict_proba(df_scaled)[0][1]
    pred = "Maladie" if proba > 0.5 else "Sain"
    
    # Affichage
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
    
    # Details
    with st.expander("Voir les details"):
        st.write("Donnees brutes:")
        st.dataframe(df)
        st.write("Donnees standardisees (ce que le modele voit):")
        st.dataframe(pd.DataFrame(df_scaled, columns=df.columns))