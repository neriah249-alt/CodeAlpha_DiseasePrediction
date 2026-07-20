# 🩺 CodeAlpha_DiseasePrediction

## Prédiction de Maladies Cardiaques — Machine Learning

Projet réalisé dans le cadre du stage **Machine Learning** chez **CodeAlpha**.

---

## 📌 Objectif

Prédire la présence de maladies cardiaques à partir de données médicales structurées en utilisant différents algorithmes de classification supervisée.

---

## 🌐 Application en ligne

🔗 **Lien de l'application** : [https://codealphadiseaseprediction-mijbvbeq7pjaapr77a97el.streamlit.app/](https://codealphadiseaseprediction-mijbvbeq7pjaapr77a97el.streamlit.app/)



---

## 📊 Dataset

- **Nom** : UCI Heart Disease Dataset
- **Source** : [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+disease)
- **Taille** : 297 patients (après nettoyage)
- **Features** : 13 variables médicales
- **Target** : Présence (1) ou absence (0) de maladie cardiaque

### Description des features

| Feature | Description | Type |
|---------|-------------|------|
| `age` | Âge du patient | Numérique |
| `sex` | Sexe (0=F, 1=M) | Catégoriel |
| `cp` | Type de douleur thoracique (1-4) | Catégoriel |
| `trestbps` | Tension artérielle au repos (mmHg) | Numérique |
| `chol` | Cholestérol sérique (mg/dl) | Numérique |
| `fbs` | Glycémie à jeun &gt; 120 mg/dl (0/1) | Binaire |
| `restecg` | Résultats ECG au repos (0-2) | Catégoriel |
| `thalach` | Fréquence cardiaque maximale | Numérique |
| `exang` | Angine induite par l'exercice (0/1) | Binaire |
| `oldpeak` | Dépression ST induite par l'exercice | Numérique |
| `slope` | Pente du segment ST (1-3) | Catégoriel |
| `ca` | Nombre de vaisseaux colorés (0-3) | Numérique |
| `thal` | Thalassémie (3=Normal, 6=Défaut fixe, 7=Défaut réversible) | Catégoriel |

---

## 🤖 Modèles testés

| Modèle | Description |
|--------|-------------|
| **Logistic Regression** | Régression logistique avec régularisation L2 |
| **SVM** | Support Vector Machine avec noyau RBF ⭐ **Meilleur modèle** |
| **Random Forest** | Forêt aléatoire (100 estimateurs) |
| **Decision Tree** | Arbre de décision simple |

---

## 📈 Résultats

### Meilleur modèle : SVM

| Métrique | Score |
|----------|-------|
| **Accuracy** | 85.00% |
| **Precision** | 88.00% |
| **Recall** | 78.57% |
| **F1-Score** | 83.02% |
| **ROC-AUC** | 95.42% |

### Comparaison des modèles

| Modèle | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|----------|---------|
| SVM | **0.8500** | **0.8800** | 0.7857 | **0.8302** | **0.9542** |
| Random Forest | **0.8500** | **0.8800** | 0.7857 | **0.8302** | 0.9408 |
| Logistic Regression | 0.8333 | 0.8462 | **0.7857** | 0.8148 | 0.9498 |
| Decision Tree | 0.7000 | 0.6923 | 0.6429 | 0.6667 | 0.6964 |

---

## 🔍 Features les plus importantes

| Rang | Feature | Importance | Description |
|------|---------|------------|-------------|
| 1 | **cp** | 14.20% | Type de douleur thoracique |
| 2 | **thal** | 12.16% | Thalassémie |
| 3 | **thalach** | 12.13% | Fréquence cardiaque maximale |
| 4 | **oldpeak** | 11.46% | Dépression ST induite |
| 5 | **ca** | 9.75% | Vaisseaux colorés par fluoroscopie |

---

## 🛠️ Technologies utilisées

- **Python 3.11**
- **Pandas** — Manipulation des données
- **NumPy** — Calculs numériques
- **Scikit-learn** — Modèles de ML et métriques
- **Matplotlib & Seaborn** — Visualisation
- **Streamlit** — Application web interactive

---

## 📁 Structure du projet
```texte
CodeAlpha_DiseasePrediction/
│
├── README.md                 # Ce fichier
├── requirements.txt          # Dépendances Python
├── main.py                   # Script principal (pipeline complet)
├── app.py                    # Application Streamlit interactive
│
├── src/
│   ├── preprocess.py         # Chargement et nettoyage des données
│   ├── train.py              # Entraînement des modèles
│   ├── evaluate.py           # Visualisations et évaluation
│   └── predict.py            # Prédiction sur nouveaux patients
│
├── data/
│   └── (dataset optionnel)
│
├── results/
│   ├── eda_heart_disease.png
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   └── feature_importance.png
│
└── models/
├── best_model.pkl        # Modèle SVM sauvegardé
└── scaler.pkl            # Scaler pour standardisation
```
---

## 🚀 Installation et utilisation

### 1. Cloner le repository

```bash
git clone https://github.com/neriah249-alt/CodeAlpha_DiseasePrediction.git
cd CodeAlpha_DiseasePrediction

2. Installer les dépendances

```bash
pip install -r requirements.txt

3. Exécuter le pipeline complet
bash
python main.py
4. Lancer l'application Streamlit (local)

bash
streamlit run app.py

L'application s'ouvrira automatiquement dans votre navigateur à l''adresse http://localhost:8501.

📊 Visualisations générées
Feuilles de calcul
| Fichier                          | Description                             |
| -------------------------------- | --------------------------------------- |
| `results/eda_heart_disease.png`  | Analyse exploratoire des données        |
| `results/model_comparison.png`   | Comparaison des métriques des modèles   |
| `results/confusion_matrix.png`   | Matrice de confusion du meilleur modèle |
| `results/roc_curves.png`         | Courbes ROC comparées                   |
| `results/feature_importance.png` | Importance des variables                |

🎯 Points clés du projet
✅ Prétraitement robuste : Gestion des valeurs manquantes, standardisation
✅ Multiple modèles : Comparaison de 4 algorithmes de classification
✅ Métriques complètes : Accuracy, Precision, Recall, F1-Score, ROC-AUC
✅ Visualisations : EDA, matrices de confusion, courbes ROC
✅ Application interactive : Interface Streamlit avec validation des entrées
✅ Détection des cas extrêmes : Alertes pour les valeurs anormales
✅ Déploiement en ligne : Accessible via Streamlit Cloud

📝 Auteur
Nom : OLAFA Maurica Nériah Mondjissiola
Stage : CodeAlpha Machine Learning Internship
LinkedIn : Mauricia Olafa
Date : 20 Juillet 2026
🙏 Remerciements
Merci à CodeAlpha pour cette opportunité de stage et l'accompagnement tout au long du projet.

<div align="center">
  <p>🎓 Projet réalisé dans le cadre du stage Machine Learning — CodeAlpha</p>
  <p>Made with ❤️ and Python</p>
</div>
```