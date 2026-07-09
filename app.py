#!/usr/bin/env python3
"""
app.py
Application Streamlit interactive - VRAI MODELE
Lancer avec : streamlit run app.py
"""

import streamlit as st
import pandas as pd
import pickle
import sys
import os

# Ajouter src/ au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from preprocess import load_data, clean_data, split_and_scale

st.set_page_config(
    page_title="Prediction de Maladies Cardiaques",
    page_icon="🩺",
    layout="centered"
)

# CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #e74c3c;
        text-align: center;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .risk-high { background-color: #ffcccc; color: #c0392b; }
    .risk-moderate { background-color: #fff3cd; color: #856404; }
    .risk-low { background-color: #d4edda; color: #155724; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_scaler():
    """Charge le modele sauvegarde et recree le scaler."""
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'best_model.pkl')
    
    with open(model_path, 'rb') as f:
        model_dict = pickle.load(f)
    
    # Recreer le scaler
    df = clean_data(load_data())
    _, _, _, _, scaler, _, _ = split_and_scale(df)
    
    return model_dict, scaler

# Chargement
try:
    model_dict, scaler = load_model_and_scaler()
    model = model_dict['model']
    model_name = model_dict['name']
    model_loaded = True
except Exception as e:
    st.error(f"Erreur chargement modele : {e}")
    model_loaded = False
    model_name = "Non charge"

# Interface
st.markdown('<p class="main-header">🩺 Prediction de Maladies Cardiaques</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Application ML — CodeAlpha Internship</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Informations")
if model_loaded:
    st.sidebar.success(f"Modele charge : **{model_name}**")
else:
    st.sidebar.error("Modele non charge")

st.sidebar.markdown("---")
st.sidebar.markdown("**Guide des features :**")
st.sidebar.markdown("""
- **cp** : Douleur thoracique (1-4)
- **thalach** : Frequence cardiaque max
- **oldpeak** : Depression ST
- **ca** : Vaisseaux colores (0-3)
- **thal** : Thalassemie (3=Normal, 6=Defaut fixe, 7=Defaut reversible)
""")

# FORMULAIRE avec st.form() pour forcer le recalcul
with st.form("patient_form"):
    st.subheader("Informations du patient")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", min_value=20, max_value=100, value=55, step=1, key="age")
        sex = st.selectbox("Sexe", [0, 1], format_func=lambda x: "Femme" if x == 0 else "Homme", key="sex")
        cp = st.selectbox("Douleur thoracique", [1, 2, 3, 4], 
                          format_func=lambda x: {1: "Typique", 2: "Atypique", 3: "Non-angineuse", 4: "Asymptomatique"}[x], key="cp")
        trestbps = st.number_input("Tension arterielle (mmHg)", min_value=90, max_value=200, value=130, step=1, key="trestbps")
    
    with col2:
        chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=240, step=1, key="chol")
        fbs = st.selectbox("Glycemie > 120", [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui", key="fbs")
        restecg = st.selectbox("ECG", [0, 1, 2], format_func=lambda x: {0: "Normal", 1: "Anomalie ST-T", 2: "Hypertrophie"}[x], key="restecg")
        thalach = st.number_input("Freq. cardiaque max", min_value=70, max_value=220, value=150, step=1, key="thalach")
    
    with col3:
        exang = st.selectbox("Angine induite", [0, 1], format_func=lambda x: "Non" if x == 0 else "Oui", key="exang")
        oldpeak = st.number_input("Depression ST", min_value=0.0, max_value=6.0, value=1.5, step=0.1, key="oldpeak")
        slope = st.selectbox("Pente ST", [1, 2, 3], format_func=lambda x: {1: "Montante", 2: "Plate", 3: "Descendante"}[x], key="slope")
        ca = st.number_input("Vaisseaux colores", min_value=0, max_value=3, value=0, step=1, key="ca")
        thal = st.selectbox("Thalassemie", [3, 6, 7], format_func=lambda x: {3: "Normal", 6: "Defaut fixe", 7: "Defaut reversible"}[x], key="thal")
    
    # Bouton dans le form - FORCE le recalcul
    submitted = st.form_submit_button("Predire", type="primary")

# Affichage du resultat SEULEMENT apres clic sur le bouton
if submitted:
    if not model_loaded:
        st.error("Le modele n'est pas charge. Verifiez que 'models/best_model.pkl' existe.")
    else:
        # Creer le DataFrame avec les valeurs actuelles
        patient_data = pd.DataFrame([{
            'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal
        }])
        
        # Afficher les donnees pour verification
        st.write("Donnees envoyees au modele :")
        st.dataframe(patient_data)
        
        # Standardiser si necessaire
        if model_name in ['SVM', 'Logistic Regression']:
            patient_data_scaled = scaler.transform(patient_data)
            proba = model.predict_proba(patient_data_scaled)[0][1]
        else:
            proba = model.predict_proba(patient_data)[0][1]
        
        prediction = 1 if proba > 0.5 else 0
        
        # Affichage resultat
        st.markdown("---")
        
        if proba > 0.7:
            risk_class = "risk-high"
            risk_text = "RISQUE ELEVE"
            recommendation = "Consultation medicale urgente recommandee."
        elif proba > 0.4:
            risk_class = "risk-moderate"
            risk_text = "RISQUE MODERE"
            recommendation = "Surveillance recommandee. Discutez avec votre medecin."
        else:
            risk_class = "risk-low"
            risk_text = "RISQUE FAIBLE"
            recommendation = "Risque faible. Continuez a maintenir un mode de vie sain."
        
        st.markdown(f'<div class="result-box {risk_class}">{risk_text}</div>', unsafe_allow_html=True)
        
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.metric("Probabilite", f"{proba*100:.1f}%")
        with col_r2:
            st.metric("Prediction", "Maladie" if prediction == 1 else "Sain")
        with col_r3:
            st.metric("Modele", model_name)
        
        st.progress(float(proba))
        
        st.markdown("---")
        st.subheader("Recommandation")
        if proba > 0.7:
            st.error(recommendation)
        elif proba > 0.4:
            st.warning(recommendation)
        else:
            st.success(recommendation)

st.markdown("---")
st.caption("Projet CodeAlpha — Machine Learning Internship")