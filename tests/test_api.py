#pip install pytest httpx
import pytest
from fastapi.testclient import TestClient
import pandas as pd
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# Ajouter le chemin racine du projet pour l'importation des modules
current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_path, '..'))
sys.path.append(project_root)

from api.api import app  # Importer l'application

# Crée une instance du client de test
client = TestClient(app)

@pytest.fixture(scope="module")
def load_data():
    # Charge le CSV avec un chemin absolu
    data_path = os.path.join(project_root, 'data/processed/test_feature_engineering_sample.csv')
    data_clients = pd.read_csv(data_path)
    data_clients['SK_ID_CURR'] = data_clients['SK_ID_CURR'].astype(float)
    return data_clients

# Tests API ici...


### Tests Unitaires ###

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

def test_get_client_data_invalid(load_data):
    response = client.get("/client_data/999999")  # ID qui n'existe pas
    assert response.status_code == 404
    assert response.json()["detail"] == "Client 999999 non trouvé."

def test_get_prediction_valid(load_data):
    valid_client_id = load_data.iloc[0]["SK_ID_CURR"]
    response = client.get(f"/prediction/{valid_client_id}")
    assert response.status_code == 200
    assert "score" in response.json()
    assert "threshold" in response.json()

def test_get_prediction_invalid(load_data):
    response = client.get("/prediction/999999")  # ID qui n'existe pas
    assert response.status_code == 404
    assert response.json()["detail"] == "Client non trouvé"

def test_get_shap_values_valid(load_data):
    valid_client_id = load_data.iloc[0]["SK_ID_CURR"]
    response = client.get(f"/shap_values/{valid_client_id}")
    assert response.status_code == 200
    assert "shap_values" in response.json()

def test_get_shap_values_invalid(load_data):
    response = client.get("/shap_values/999999")  # ID qui n'existe pas
    assert response.status_code == 404
    assert response.json()["detail"] == "Client non trouvé"

### Tests Variés ###

def test_internal_server_error(load_data):
    pass  # Placeholder pour implémentation future





#execution des tests:
#pytest test_api.py
"""Tests de récupération des données client :

test_get_all_client_ids et test_get_all_client_data vérifient si l'API retourne correctement toutes les ID de clients et toutes les données des clients.
Tests de récupération des données d'un client spécifique :

test_get_client_data_valid et test_get_client_data_invalid testent la récupération de données pour un client valide et invalide respectivement.
Tests de prédiction :

test_get_prediction_valid et test_get_prediction_invalid valident que l'API renvoie une prédiction correcte pour un client valide et renvoie une erreur pour un client invalide.
Tests de valeurs SHAP :

test_get_shap_values_valid et test_get_shap_values_invalid testent le bon fonctionnement de la récupération des valeurs SHAP pour des clients valides et invalides.
Tests pour les exceptions :

test_internal_server_error vérifie si l'API gère correctement une situation d'erreur interne. Il est nécessaire d'adapter l'URL ou la logique pour provoquer cette erreur."""