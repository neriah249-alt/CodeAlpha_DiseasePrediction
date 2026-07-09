import sys
sys.path.insert(0, 'src')

from predict import predict_disease, load_model
from preprocess import split_and_scale, clean_data, load_data

# Charger le modèle et le scaler
df = clean_data(load_data())
_, _, _, _, scaler, _, _ = split_and_scale(df)
model_dict = load_model('models/best_model.pkl')

# TON PATIENT
mon_patient = { 
    'age': 20,    #age
    'sex': 0,   # 0=femme, 1=homme
    'cp': 3,     #douleur thoracique(1-4)
    'trestbps': 140,  #Tension artérielle
    'chol': 260,  #Cholestérol
    'fbs': 0, # Glycémie supérieur à 120 (0/1)
    'restecg': 1,   #ECG (0-2)
    'thalach': 145,   #Fréquence cardiaque maximale
    'exang': 1,  #Angine adulte (0/1)
    'oldpeak': 2.5,  #Dépression ST
    'slope': 2,  #Pente ST
    'ca': 1,   #Vaisseaux colorés
    'thal': 6   #Thalassémie (3, 6, 7)
}

# Prédire
result = predict_disease(mon_patient, model_dict, scaler)

print("=" * 50)
print("🩺 RÉSULTAT POUR TON PATIENT")
print("=" * 50)
print(f"Prédiction : {result['prediction']}")
print(f"Probabilité : {result['probabilité']}%")
print(f"Modèle : {result['model_utilisé']}")