import pytest
from fastapi.testclient import TestClient
import pandas as pd
import os
import sys
import warnings
import joblib
import traceback

import os
os.environ["NUMBA_DISABLE_JIT"] = "1"


warnings.filterwarnings("ignore", category=DeprecationWarning)

# Ajouter le chemin racine du projet pour l'importation des modules
current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_path, '..'))
sys.path.append(project_root)

from api.api import app  # Importer l'application

# Instance du client de test pour l'API
client = TestClient(app)

# Définir le chemin du modèle
pipeline_path = os.path.join(project_root, 'models', 'xgb_pipeline_tuned.pkl')

@pytest.fixture(scope="module")
def load_model():
    try:
        model = joblib.load(pipeline_path)
        return model
    except Exception as e:
        pytest.fail(f"Échec du chargement du modèle : {str(e)}")

@pytest.fixture(scope="module")
def load_data():
    try:
        data_path = os.path.join(project_root, 'data/processed/test_feature_engineering_sample.csv')
        data_clients = pd.read_csv(data_path)
        data_clients['SK_ID_CURR'] = data_clients['SK_ID_CURR'].astype(float)
        return data_clients
    except Exception as e:
        pytest.fail(f"Échec du chargement des données : {str(e)}")

        

### Tests Unitaires ###

def test_model_loading(load_model):
    assert load_model is not None, "Le modèle n'a pas été chargé correctement"

def test_get_all_client_ids(load_data):
    response = client.get("/client_data/all")
    assert response.status_code == 200
    assert len(response.json()["data"]) == len(load_data["SK_ID_CURR"].unique())

def test_get_all_client_data(load_data):
    response = client.get("/client_data/all_full")
    assert response.status_code == 200
    assert len(response.json()["data"]) == len(load_data)

def test_get_client_data_valid(load_data):
    valid_client_id = load_data.iloc[0]["SK_ID_CURR"]
    response = client.get(f"/client_data/{valid_client_id}")
    assert response.status_code == 200
    assert response.json()["data"][0]["SK_ID_CURR"] == valid_client_id

def test_get_client_data_invalid():
    response = client.get("/client_data/999999")  # ID qui n'existe pas
    assert response.status_code == 404
    assert response.json()["detail"] == "Client 999999 non trouvé."


def test_direct_prediction(load_model, load_data):
    try:
        # Vérifiez simplement que le modèle est chargé
        assert load_model is not None, "Le modèle n'a pas pu être chargé"
        print("Modèle chargé avec succès")
        
        # Tester une prédiction simple avec un sous-ensemble minimal de données
        client_data = load_data[load_data['SK_ID_CURR'] == 218796.0].drop(columns=["TARGET"], errors="ignore")
        client_data_np = client_data.to_numpy()[:1]  # Utilisez uniquement une ligne pour tester
        
        prediction_proba = load_model.predict_proba(client_data_np)[:, 1][0]
        assert prediction_proba is not None, "Erreur dans la prédiction"
    except Exception as e:
        pytest.fail(f"Erreur dans la prédiction directe : {str(e)}")
