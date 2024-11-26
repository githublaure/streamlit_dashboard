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

# Add the project root path for module import
current_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_path, '..'))
sys.path.append(project_root)

from api.api import app  # Import the application

# Test client instance for the API
client = TestClient(app)

# Define the model path
pipeline_path = os.path.join(project_root, 'models', 'xgb_pipeline_tuned.pkl')

@pytest.fixture(scope="module")
def load_model():
    try:
        model = joblib.load(pipeline_path)
        return model
    except Exception as e:
        pytest.fail(f"Failed to load the model: {str(e)}")

@pytest.fixture(scope="module")
def load_data():
    try:
        data_path = os.path.join(project_root, 'data/processed/test_feature_engineering_sample.csv')
        data_clients = pd.read_csv(data_path)
        data_clients['SK_ID_CURR'] = data_clients['SK_ID_CURR'].astype(float)
        return data_clients
    except Exception as e:
        pytest.fail(f"Failed to load the data: {str(e)}")

        

### Unit Tests ###

def test_model_loading(load_model):
    assert load_model is not None, "The model failed to load correctly"

def test_get_all_client_ids(load_data):
    response = client.get("/client_data/all")
    assert response.status_code == 200
    assert len(response.json()["data"]) == len(load_data["SK_ID_CURR"].unique())

def test_get_all_client_data():
    response = client.get("/client_data/all_full")
    assert response.status_code == 200
    # Vérifier que la route retourne exactement 10 clients
    assert len(response.json()["data"]) == 10, "The route does not return the expected 10 rows"


def test_get_client_data_valid(load_data):
    valid_client_id = load_data.iloc[0]["SK_ID_CURR"]
    response = client.get(f"/client_data/{valid_client_id}")
    assert response.status_code == 200
    assert response.json()["data"][0]["SK_ID_CURR"] == valid_client_id

"""def test_get_client_data_invalid():
    response = client.get("/client_data/999999")  # Non-existent ID
    assert response.status_code == 404
    assert response.json()["detail"] == "Client 999999 not found.""""


def test_direct_prediction(load_model, load_data):
    try:
        # Simply check if the model is loaded
        assert load_model is not None, "The model failed to load"
        print("Model loaded successfully")
        
        # Test a simple prediction with a minimal subset of data
        client_data = load_data[load_data['SK_ID_CURR'] == 218796.0].drop(columns=["TARGET"], errors="ignore")
        client_data_np = client_data.to_numpy()[:1]  # Use only one row for testing
        
        prediction_proba = load_model.predict_proba(client_data_np)[:, 1][0]
        assert prediction_proba is not None, "Error in prediction"
    except Exception as e:
        pytest.fail(f"Error in direct prediction: {str(e)}")
