#!/usr/bin/env python3
"""
app.py - VERSION AMÉLIORÉE AVEC DÉTECTION DES CAS EXTRÊMES
"""

import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Prediction Cardiaque", page_icon="🩺")

st.title("🩺 Prédiction de Maladies Cardiaques")
st.write("Application ML — CodeAlpha Internship")

# ==========================================
# CHARGEMENT DU MODÈLE
# ==========================================
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

# ==========================================
# SEUILS DE RISQUE AJUSTABLES
# ==========================================
# Ces seuils sont plus larges pour mieux différencier les cas
RISK_LOW = 0.30   # En dessous = Faible
RISK_HIGH = 0.65  # Au-dessus = Élevé

# ==========================================
# DÉTECTION DES VALEURS EXTRÊMES
# ==========================================
def detect_extreme_values(age, chol, trestbps, oldpeak, thalach):
    """Détecte les valeurs extrêmes et retourne une alerte"""
    alerts = []
    risk_score = 0
    
    # Âge avancé
    if age > 70:
        alerts.append(f"🔴 Âge avancé ({age} ans) → facteur de risque majeur")
        risk_score += 2
    elif age > 60:
        alerts.append(f"🟡 Âge ({age} ans) → risque modéré")
        risk_score += 1
    
    # Cholestérol très élevé
    if chol > 300:
        alerts.append(f"🔴 Cholestérol très élevé ({chol} mg/dL) → risque majeur")
        risk_score += 2
    elif chol > 240:
        alerts.append(f"🟡 Cholestérol élevé ({chol} mg/dL)")
        risk_score += 1
    
    # Tension artérielle élevée
    if trestbps > 160:
        alerts.append(f"🔴 Tension très élevée ({trestbps} mmHg)")
        risk_score += 2
    elif trestbps > 140:
        alerts.append(f"🟡 Tension élevée ({trestbps} mmHg)")
        risk_score += 1
    
    # Oldpeak élevé (dépression ST)
    if oldpeak > 4:
        alerts.append(f"🔴 Dépression ST sévère ({oldpeak})")
        risk_score += 2
    elif oldpeak > 2.5:
        alerts.append(f"🟡 Dépression ST significative ({oldpeak})")
        risk_score += 1
    
    # Fréquence cardiaque maximale basse (mauvais signe)
    if thalach < 100:
        alerts.append(f"🔴 FC max très basse ({thalach} bpm)")
        risk_score += 2
    elif thalach < 120:
        alerts.append(f"🟡 FC max basse ({thalach} bpm)")
        risk_score += 1
    
    return alerts, risk_score

# ==========================================
# FONCTION DE PRÉDICTION AVEC AJUSTEMENT
# ==========================================
def predict_with_risk_adjustment(df, model, scaler):
    """
    Prédiction avec ajustement basé sur les valeurs extrêmes
    """
    # Prédiction standard
    df_scaled = scaler.transform(df)
    proba = model.predict_proba(df_scaled)[0][1]
    
    # Récupération des valeurs pour détection d'extrêmes
    row = df.iloc[0]
    alerts, risk_score = detect_extreme_values(
        row['age'], row['chol'], row['trestbps'], 
        row['oldpeak'], row['thalach']
    )
    
    # Ajustement de la probabilité en fonction du score de risque extrême
    # +5% par niveau de risque extrême (max +20%)
    adjusted_proba = proba + (risk_score * 0.03)
    adjusted_proba = min(adjusted_proba, 0.99)  # Plafonnement
    
    # Niveau de risque avec SEUILS ÉLARGIS
    if adjusted_proba < RISK_LOW:
        level = "Faible"
        emoji = "🟢"
        color = "success"
        detail = "Risque faible. Continuez à maintenir un mode de vie sain."
    elif adjusted_proba < RISK_HIGH:
        level = "Moyen"
        emoji = "🟡"
        color = "warning"
        detail = "Risque modéré. Une consultation médicale est recommandée."
    else:
        level = "Élevé"
        emoji = "🔴"
        color = "error"
        detail = "⚠️ Risque élevé ! Consultez un médecin immédiatement."
    
    return adjusted_proba, level, emoji, color, detail, alerts

# ==========================================
# FORMULAIRE
# ==========================================
with st.form("form"):
    st.subheader("📋 Informations du patient")
    
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
    
    submit = st.form_submit_button("🔍 Prédire", type="primary")

# ==========================================
# PRÉDICTION ET AFFICHAGE
# ==========================================
if submit and model_ok:
    # Préparation des données
    df = pd.DataFrame([{
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
        'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
        'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }])
    
    # Prédiction avec ajustement
    proba, level, emoji, color, detail, alerts = predict_with_risk_adjustment(
        df, model, scaler
    )
    
    # ==========================================
    # AFFICHAGE DES RÉSULTATS
    # ==========================================
    st.markdown("---")
    st.subheader("📈 Résultat de la prédiction")
    
    # Carte de résultat colorée
    color_map = {
        "success": "#4CAF50",
        "warning": "#FFC107",
        "error": "#F44336"
    }
    bg_color = color_map.get(color, "#4CAF50")
    
    result_html = f"""
    <div style="padding: 25px; border-radius: 15px; background-color: {bg_color}20; 
                border: 3px solid {bg_color}; text-align: center;">
        <h1 style="color: {bg_color}; font-size: 3em;">{emoji} Niveau de risque : {level}</h1>
        <h2 style="color: {bg_color};">Probabilité : {proba*100:.1f}%</h2>
        <p style="font-size: 1.1em;">{detail}</p>
        <hr>
        <p style="font-size: 0.9em; color: #666;">
            <i>Seuils : {RISK_LOW:.0%} (faible) | {RISK_HIGH:.0%} (élevé)</i>
        </p>
    </div>
    """
    st.markdown(result_html, unsafe_allow_html=True)
    
    # ==========================================
    # AFFICHAGE DES ALERTES
    # ==========================================
    if alerts:
        st.warning("⚠️ **Facteurs de risque détectés :**")
        for alert in alerts:
            st.write(f"- {alert}")
    else:
        st.success("✅ Aucun facteur de risque extrême détecté.")
    
    # ==========================================
    # MÉTRIQUES DÉTAILLÉES
    # ==========================================
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Probabilité", f"{proba*100:.1f}%")
    with col2:
        st.metric("Niveau", level)
    with col3:
        st.metric("Modèle", model_name)
    with col4:
        st.metric("Facteurs extrêmes", len(alerts))
    
    # Barre de progression
    st.progress(float(proba))
    
    # ==========================================
    # JAUGE DE RISQUE VISUELLE
    # ==========================================
    st.subheader("📊 Jauge de risque")
    
    fig, ax = plt.subplots(figsize=(10, 1.5))
    
    # Dégradé de couleurs
    colors = ['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336']
    for i in range(100):
        x = i / 100
        if x < 0.3:
            c = colors[0]
        elif x < 0.5:
            c = colors[1]
        elif x < 0.65:
            c = colors[2]
        elif x < 0.8:
            c = colors[3]
        else:
            c = colors[4]
        ax.barh(0, 0.01, left=x, height=0.5, color=c, edgecolor='none')
    
    # Marqueur de la probabilité
    ax.scatter(proba, 0, color='black', s=250, zorder=5, marker='v')
    ax.axvline(proba, color='black', linestyle='--', alpha=0.5)
    
    # Étiquettes
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_color('#ccc')
    
    # Ajout des zones de risque
    ax.text(0.15, -0.45, 'Faible', ha='center', fontsize=10, color='#4CAF50')
    ax.text(0.40, -0.45, 'Modéré', ha='center', fontsize=10, color='#FFC107')
    ax.text(0.80, -0.45, 'Élevé', ha='center', fontsize=10, color='#F44336')
    
    # Ligne de la probabilité
    ax.text(proba, -0.7, f'{proba*100:.0f}%', ha='center', fontsize=14, fontweight='bold')
    
    st.pyplot(fig)
    
    # ==========================================
    # INTERPRÉTATION
    # ==========================================
    with st.expander("📖 Interprétation détaillée"):
        st.markdown(f"""
        **Facteurs de risque :**
        
        | Facteur | Valeur | Statut |
        |---------|--------|--------|
        | Âge | {age} ans | {'⚠️ > 60 ans' if age > 60 else '✅ Normal'} |
        | Cholestérol | {chol} mg/dL | {'⚠️ > 240' if chol > 240 else '✅ Normal'} |
        | Tension | {trestbps} mmHg | {'⚠️ > 140' if trestbps > 140 else '✅ Normal'} |
        | FC max | {thalach} bpm | {'⚠️ < 120' if thalach < 120 else '✅ Normal'} |
        | Dépression ST | {oldpeak} | {'⚠️ > 2.5' if oldpeak > 2.5 else '✅ Normal'} |
        | Douleur thoracique | {cp} | {'⚠️ Asymptomatique' if cp == 4 else '✅ Normal'} |
        
        **Recommandation :** {detail}
        """)

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("---")
    st.subheader("ℹ️ À propos")
    st.markdown("""
    **Dataset :** Heart Disease (UCI Cleveland)
    
    **Modèle :** {}
    
    **Caractéristiques :** 13 features
    
    **Technologies :**
    - Streamlit
    - Scikit-learn
    - Pickle
    """.format(model_name if model_ok else "Non chargé"))
    
    st.markdown("---")
    st.caption("🎓 CodeAlpha Internship 2026 | Tâche 4")

st.markdown("---")
st.caption("⚠️ Application éducative — Consultez toujours un professionnel de santé")