# 🩺 CodeAlpha_DiseasePrediction

## Prédiction de Maladies Cardiaques — Machine Learning

Projet réalisé dans le cadre du stage **Machine Learning** chez **CodeAlpha**.

---

## 📌 Objectif

Prédire la présence de maladies cardiaques à partir de données médicales structurées en utilisant différents algorithmes de classification supervisée.

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

<<<<<<< HEAD
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
└── best_model.pkl        # Modèle SVM sauvegardé
=======
>>>>>>> 31f7643781b01842a92266accdfb6d01146a48c1
