from fastapi import FastAPI, HTTPException, Depends
import joblib
import pandas as pd
import numpy as np
import logging


app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Charger le pipeline complet (prétraitement + modèle)
pipeline_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'xgb_pipeline_tuned.pkl')
loaded_pipeline = joblib.load(pipeline_path)
explainer_path = "../models/shap_explainer.pkl"

# Variable globale pour stocker l'explainer et éviter de le recharger à chaque requête
loaded_shap_explainer = None

def get_explainer():
    global loaded_shap_explainer
    if loaded_shap_explainer is None:
        loaded_shap_explainer = joblib.load(explainer_path)
    return loaded_shap_explainer



# Fonction pour charger les données des clients
def load_data():
    data_clients = pd.read_csv("../data/processed/test_feature_engineering_sample.csv")
    data_clients['SK_ID_CURR'] = data_clients['SK_ID_CURR'].astype(float)
    return data_clients

# Fonction pour obtenir les données des clients (dépendance)
def get_data_clients():
    return load_data()  # Charge les données via la fonction load_data

@app.get("/client_data/all")
def get_all_client_ids(data_clients: pd.DataFrame = Depends(get_data_clients)):
    try:
        client_ids = data_clients["SK_ID_CURR"].unique().tolist()
        return {"data": [{"SK_ID_CURR": client_id} for client_id in client_ids]}
    except Exception as e:
        logger.error(f"Error in retrieving all client IDs: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur : {str(e)}")

@app.get("/client_data/all_full")
def get_all_client_data(data_clients: pd.DataFrame = Depends(get_data_clients)):
    try:
        # Convertir explicitement les données pour éviter des problèmes de sérialisation
        data_clients_json_ready = data_clients.astype(object).where(pd.notnull(data_clients), None)
        return {"data": data_clients_json_ready.to_dict(orient="records")}
    except Exception as e:
        logger.error(f"Error in retrieving all client data: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur : {str(e)}")

@app.get("/client_data/{client_id}") #202006.0
def get_client_data(client_id: int, data_clients: pd.DataFrame = Depends(get_data_clients)):
    try:
        client_id_as_float = float(client_id)
        client_data = data_clients[data_clients["SK_ID_CURR"] == client_id_as_float]

        if client_data.empty:
            raise HTTPException(status_code=404, detail=f"Client {client_id} non trouvé.")

        client_data = client_data.replace({np.nan: None})
        return {"data": client_data.to_dict(orient="records")}
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in retrieving client data: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur : {str(e)}")

@app.get("/prediction/{client_id}")
def get_prediction(client_id: int, data_clients: pd.DataFrame = Depends(get_data_clients)):
    try:
        logger.info(f"Attempting prediction for client ID: {client_id}")
        client_id_as_float = float(client_id)
        client_data = data_clients[data_clients["SK_ID_CURR"] == client_id_as_float].drop(columns=["TARGET"], errors='ignore')

        if client_data.empty:
            logger.warning(f"Client {client_id} not found in data")
            raise HTTPException(status_code=404, detail="Client non trouvé")

        logger.info(f"Client data shape: {client_data.shape}")
        client_data_np = client_data.to_numpy()
        logger.info("Making prediction")
        prediction_proba = loaded_pipeline.predict_proba(client_data_np)[:, 1][0]
        threshold = getattr(loaded_pipeline.named_steps['model'], 'get_threshold', lambda: 0.5)()
        logger.info(f"Prediction made: {prediction_proba}, Threshold: {threshold}")

        return {"score": float(prediction_proba), "threshold": float(threshold)}
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in prediction: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur : {str(e)}")



@app.get("/shap_values/{client_id}")
def get_shap_values(client_id: int, data_clients: pd.DataFrame = Depends(get_data_clients)):
    try:
        client_id_as_float = float(client_id)
        client_data = data_clients[data_clients["SK_ID_CURR"] == client_id_as_float].drop(columns=["TARGET"], errors='ignore')

        if client_data.empty:
            raise HTTPException(status_code=404, detail="Client non trouvé")

        explainer = get_explainer()
        client_data_np = client_data.to_numpy().reshape(1, -1)
        client_shap_values = explainer.shap_values(client_data_np)

        # Formatage des valeurs SHAP
        if isinstance(client_shap_values, list) and len(client_shap_values) == 1:
            client_shap_values = client_shap_values[0]

        return {"shap_values": client_shap_values.tolist()}
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des valeurs SHAP : {e}")
        raise HTTPException(status_code=500, detail="Erreur interne du serveur")

