from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np

# Charger le pipeline complet (prétraitement + modèle)
pipeline_path = "../models/xgb_pipeline_tuned.pkl"
loaded_pipeline = joblib.load(pipeline_path)

# Charger l'explainer SHAP
explainer_path = "../models/shap_explainer.pkl"
loaded_shap_explainer = joblib.load(explainer_path)

# Créer l'API
app = FastAPI()

# Charger les données des clients
data_clients = pd.read_csv("../data/processed/test_feature_engineering_sample.csv")

# Assurez-vous que la colonne SK_ID_CURR est bien en float
data_clients['SK_ID_CURR'] = data_clients['SK_ID_CURR'].astype(float)

# Debugging: Vérifiez si les données sont bien chargées
print(f"Data loaded successfully with {len(data_clients)} rows.")

from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np

# Charger le pipeline complet (prétraitement + modèle)
pipeline_path = "../models/xgb_pipeline_tuned.pkl"
loaded_pipeline = joblib.load(pipeline_path)

# Charger l'explainer SHAP
explainer_path = "../models/shap_explainer.pkl"
loaded_shap_explainer = joblib.load(explainer_path)

# Créer l'API
app = FastAPI()


# Charger les données des clients
data_clients = pd.read_csv("../data/processed/test_feature_engineering_sample.csv")

# Assurez-vous que la colonne SK_ID_CURR est bien en float
data_clients['SK_ID_CURR'] = data_clients['SK_ID_CURR'].astype(float)

# Fonction pour récupérer les données du client

@app.get("/client_data/all")
def get_all_client_ids():
    try:
        # Retourner uniquement la colonne SK_ID_CURR pour tous les clients disponibles
        client_ids = data_clients["SK_ID_CURR"].unique().tolist()
        return {"data": [{"SK_ID_CURR": client_id} for client_id in client_ids]}
    
    except Exception as e:
        print(f"Error in retrieving client IDs: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur : {str(e)}")
    
@app.get("/client_data/all_full")
def get_all_client_data():
    try:
        # Convertir explicitement les données pour éviter des problèmes de sérialisation
        data_clients_json_ready = data_clients.astype(object).where(pd.notnull(data_clients), None)
        return {"data": data_clients_json_ready.to_dict(orient="records")}
    except Exception as e:
        print(f"Error in retrieving full client data: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur : {str(e)}")



@app.get("/client_data/{client_id}")
def get_client_data(client_id: int):
    try:
        # Convertir l'ID client en float car SK_ID_CURR est stocké en floats
        client_id_as_float = float(client_id)
        
        # S'assurer que SK_ID_CURR est bien considéré comme un float dans le dataset
        data_clients['SK_ID_CURR'] = data_clients['SK_ID_CURR'].astype(float)
        
        # Trouver les données du client
        client_data = data_clients[data_clients["SK_ID_CURR"] == client_id_as_float]
        
        # Vérifier si les données sont trouvées
        if client_data.empty:
            raise HTTPException(status_code=404, detail=f"Client {client_id} non trouvé.")
        
        # Remplacer les NaN par une chaîne vide ou une autre valeur par défaut (ici, on utilise 0 comme exemple)
        client_data = client_data.replace({np.nan: None})  # Ou remplacez par 0 si cela convient mieux
        
        # Retourner les données du client sous forme JSON
        return {"data": client_data.to_dict(orient="records")}
    
    except Exception as e:
        # Afficher un message d'erreur détaillé pour faciliter le débogage
        print(f"Error in retrieving client data: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur : {str(e)}")



# Fonction pour obtenir la prédiction du modèle pour un client
@app.get("/prediction/{client_id}")
def get_prediction(client_id: int):
    try:
        # Convert the client_id to float to match the SK_ID_CURR data type
        client_id_as_float = float(client_id)
        
        # Debugging: Check if the client ID is being passed and cast correctly
        print(f"Searching for client: {client_id_as_float}")
        
        # Récupère les données du client
        client_data = data_clients[data_clients["SK_ID_CURR"] == client_id_as_float].drop(columns=["TARGET"], errors='ignore')
        
        if client_data.empty:
            print(f"Client {client_id_as_float} not found!")
            raise HTTPException(status_code=404, detail="Client non trouvé")
        
        print(f"Found client data: {client_data.head()}")

        # Convert the DataFrame to NumPy array without feature names to avoid warning
        client_data_np = client_data.to_numpy()
        
        # Debugging: Ensure data is in the correct shape
        print(f"Client data shape: {client_data_np.shape}")

        # Obtenir les prédictions
        prediction_proba = loaded_pipeline.predict_proba(client_data_np)[:, 1][0]
        threshold = loaded_pipeline.named_steps['model'].get_threshold()

        print(f"Prediction: {prediction_proba}, Threshold: {threshold}")
        
        # Convert numpy.float32 to Python float before returning it
        return {"score": float(prediction_proba), "threshold": float(threshold)}
    
    except Exception as e:
        # Handle any unexpected errors
        print(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")


# Fonction pour obtenir les valeurs SHAP du client
@app.get("/shap_values/{client_id}")
def get_shap_values(client_id: int):
    try:
        # Convert client_id to float
        client_id_as_float = float(client_id)
        
        # Debugging: Check if the client ID is being passed and cast correctly
        print(f"Getting SHAP values for client: {client_id_as_float}")
        
        # Récupère les données du client
        client_data = data_clients[data_clients["SK_ID_CURR"] == client_id_as_float].drop(columns=["TARGET"], errors='ignore')
        
        if client_data.empty:
            print(f"Client {client_id_as_float} not found!")
            raise HTTPException(status_code=404, detail="Client non trouvé")

        # Convert client data to NumPy array
        client_data_np = client_data.to_numpy().reshape(1, -1)  # Ensure 2D input

        # Debugging: Ensure data is in the correct shape for SHAP explainer
        print(f"Client data shape for SHAP: {client_data_np.shape}")

        # Calculer les valeurs SHAP pour ce client
        client_shap_values = loaded_shap_explainer.shap_values(client_data_np)

        # Return SHAP values as JSON response
        return {"shap_values": client_shap_values.tolist()}
    
    except Exception as e:
        # Handle any unexpected errors
        print(f"Error in getting SHAP values: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
