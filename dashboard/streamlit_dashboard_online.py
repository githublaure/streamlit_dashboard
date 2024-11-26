import os
import streamlit as st
import requests
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import joblib
import numpy as np


# API URL
api_url = "http://18.233.222.214"

# Build the path to the model in a robust way
base_path = os.path.dirname(os.path.abspath(__file__))  # Get the directory of this script
explainer_path = os.path.join(base_path, '..', 'models', 'shap_explainer.pkl')

# Load the explainer
if not os.path.exists(explainer_path):
    st.error(f"The explainer file was not found at the specified path: {explainer_path}")
else:
    explainer = joblib.load(explainer_path)



# Streamlit Dashboard Title
st.title("Credit Prediction Dashboard")

# API call to collect all data from client IDs
data_clients_url = f"{api_url}/client_data/all"
try:
    response = requests.get(data_clients_url)
    response.raise_for_status()  # Check for HTTP errors
    data_clients_response = response.json()

    # Makes sure data is available
    if "data" not in data_clients_response:
        st.error("Error: The API response does not contain 'data'.")
        client_ids = []
    else:
        client_ids = [client['SK_ID_CURR'] for client in data_clients_response['data']]
except requests.exceptions.RequestException as e:
    st.error(f"Error loading client data: {str(e)}")
    client_ids = []

# Select client ID with a `selectbox`
client_id = st.selectbox("Select Client ID", client_ids)

# Saves client ID in a `session_state`
if 'client_id' not in st.session_state:
    st.session_state['client_id'] = client_id

# Call API to retrieve complete data of clients
all_client_data_url = f"{api_url}/client_data/all_full"
try:
    all_data_response = requests.get(all_client_data_url)
    all_data_response.raise_for_status()  # Check for HTTP errors
    all_data_response = all_data_response.json()
    if "data" not in all_data_response:
        st.error("Error: Complete client data is not available.")
        all_data = pd.DataFrame()
    else:
        all_data = pd.DataFrame(all_data_response['data'])  # All complete client data
except requests.exceptions.RequestException as e:
    st.error(f"Error loading complete data: {str(e)}")
    all_data = pd.DataFrame()  



# Checks if data are available
if not all_data.empty:
    # Call Api to retrieve data from a specific client
    client_data_url = f"{api_url}/client_data/{client_id}"
    client_response = requests.get(client_data_url).json()

if "detail" in client_response:
    st.error(f"Error: {client_response['detail']}")
else:
    client_data = pd.DataFrame(client_response['data'])

    # Shows client's data
    st.subheader("Client Data")
    st.write(client_data)

    # Extract each features 
    available_features = client_data.columns.tolist()

    # Select which feature to compare
    selected_feature = st.selectbox("Select Feature to Compare", available_features)

    # Compare feature with the mean of the population
    st.subheader(f"Comparison of {selected_feature} between the client and the mean")

        # Compare feature with the mean of the population
    client_value = client_data[selected_feature].values[0]

    # Check if the variable exists in all data
    if selected_feature not in all_data.columns:
        st.error(f"The column {selected_feature} does not exist in the complete data.")
    else:
        mean_value = all_data[selected_feature].mean()

        # Visualization of the comparison
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))

        # Client vs Mean Comparison
        sns.barplot(x=["client", "mean"], y=[client_value, mean_value], ax=ax[0])
        ax[0].set_title(f"Comparison of the client to the mean ({selected_feature})")

        # Distribution in the population
        sns.boxplot(x=all_data[selected_feature], ax=ax[1])
        ax[1].set_title(f"Distribution of {selected_feature} in the population")

        st.pyplot(fig)  # Display the graph

    # API call to get the prediction and threshold
    prediction_url = f"{api_url}/prediction/{client_id}"
    prediction_response = requests.get(prediction_url).json()
    prediction_score = prediction_response['score']
    threshold = prediction_response['threshold']  # The threshold retrieved from the API

    # Calculate the business score based on F-beta
    prediction_percentage = round(prediction_score * 100, 2)
    
    # Display the result as an indicator
    st.subheader("Prediction Result")
    st.markdown(f"""
        **Score Explanation:**
        - This score of **{prediction_percentage}%** is an estimate of the probability that this client **will not repay** the credit (score = 1).
        - If this score exceeds the threshold of **{round(threshold * 100, 2)}%**, the credit is **rejected**.
        - Below this threshold, the credit is **approved**.
    """)

    score_text = "Credit Rejected" if prediction_score >= threshold else "Credit Approved"

    # Gauge chart with the prediction score and threshold
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction_score * 100,  # Score in percentage
        title={'text': f"Prediction Score: {prediction_percentage}%"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, threshold * 100], 'color': "green"},
                {'range': [threshold * 100, 100], 'color': "red"},
            ],
            'threshold': {
                'line': {'color': "blue", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100  # Threshold line
            }
        }
    ))

    st.plotly_chart(fig_gauge)

    # Display the final decision
    st.markdown(f"**Decision:** {score_text} (Threshold: {round(threshold * 100, 2)}%)")

    # API call to get the local SHAP values
    shap_values_url = f"{api_url}/shap_values/{client_id}"
    shap_response = requests.get(shap_values_url).json()
    shap_values = shap_response['shap_values']

    # Correction to avoid error with lists
    shap_values_np = np.array(shap_values[0])  # Convert to NumPy array

    # Make sure `base_values` is a single float (for binary or multi-class models)
    if isinstance(explainer.expected_value, np.ndarray) and len(explainer.expected_value) > 1:
        base_value = explainer.expected_value[0]  # For multi-class model
    else:
        base_value = float(explainer.expected_value)  # Binary model

    # Display the expected value
    st.subheader("Expected Value")
    st.markdown(f"The expected value of the model is: **{round(base_value, 4)}**.")
    st.markdown("This is the average prediction of the model before taking into account the specific contributions of each variable.")

    # Waterfall plot: Local analysis of SHAP values
    st.subheader("Local Analysis of SHAP Values: Specific contribution of each feature for this client")
    st.markdown("This graph shows how different features influence the prediction for this particular client. Each bar indicates the impact of a feature that pushes the prediction either up or down.")
    fig_local = plt.figure(figsize=(10, 6))
    shap.waterfall_plot(shap.Explanation(values=shap_values_np, base_values=base_value, data=client_data.iloc[0, :]), show=False)
    st.pyplot(fig_local)  # Display the Waterfall graph with st.pyplot

    # Bar plot: Global analysis of SHAP values
    st.subheader("Global Analysis of SHAP Values: Importance of features across all clients")
    st.markdown("This graph shows which features are globally most important for the model, on average, across all clients.")
    fig_global = plt.figure(figsize=(10, 6))
    shap.summary_plot(np.array(shap_values), client_data, plot_type="bar", show=False)
    st.pyplot(fig_global)  # Display the bar graph with st.pyplot

    st.subheader("© Streamlit dashboard by Laure Agrech")