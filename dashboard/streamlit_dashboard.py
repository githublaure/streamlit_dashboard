import streamlit as st
import requests
import pandas as pd
import seaborn as sns
import numpy as np
import shap
import plotly.graph_objects as go
import matplotlib.pyplot as plt


# API URL
api_url = "http://127.0.0.1:5001"

# Dashboard Title
st.title("Credit Prediction Dashboard")


# API call to get client data
data_clients_url = f"{api_url}/client_data/all"
try:
    response = requests.get(data_clients_url)
    response.raise_for_status()
    data_clients_response = response.json()
    client_ids = [client['SK_ID_CURR'] for client in data_clients_response['data']] if "data" in data_clients_response else []
except requests.exceptions.RequestException as e:
    st.error(f"Error loading client data: {str(e)}")
    client_ids = []

# Select client ID using a selectbox
client_id = st.selectbox("Select Client ID", client_ids)

# Keep track of selected client state
if 'selected_client' not in st.session_state:
    st.session_state['selected_client'] = None

# Update selected client state when a change is made
if client_id:
    st.session_state['selected_client'] = client_id

# Check if a client is selected
if st.session_state['selected_client']:
    # API call to retrieve specific client data
    client_data_url = f"{api_url}/client_data/{st.session_state['selected_client']}"
    try:
        client_response = requests.get(client_data_url).json()
        if "detail" in client_response:
            st.error(f"Error: {client_response['detail']}")
        else:
            client_data = pd.DataFrame(client_response['data'])

            # Display client data
            st.subheader("Client Data")
            st.write(client_data)

            # Extract available features
            available_features = client_data.columns.tolist()

            # Select variable to compare
            selected_feature = st.selectbox("Select Variable to Compare", available_features)

            # Button to display variable comparison
            if st.button("Show Variable Comparison"):
                st.subheader(f"Comparison of {selected_feature} between the client and the average")

                all_client_data_url = f"{api_url}/client_data/all_full"
                all_data_response = requests.get(all_client_data_url).json()
                if "data" in all_data_response:
                    all_data = pd.DataFrame(all_data_response['data'])
                    client_value = client_data[selected_feature].values[0]

                    if selected_feature not in all_data.columns:
                        st.error(f"Column {selected_feature} does not exist in the complete data.")
                    else:
                        mean_value = all_data[selected_feature].mean()

                        # Visualization of the comparison
                        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                        sns.barplot(x=["client", "average"], y=[client_value, mean_value], ax=ax[0])
                        ax[0].set_title(f"Comparison of the client to the average ({selected_feature})")

                        sns.boxplot(x=all_data[selected_feature], ax=ax[1])
                        ax[1].set_title(f"Distribution of {selected_feature} in the population")

                        st.pyplot(fig)

            # Button to display prediction score and gauge
            if st.button("Show Prediction Score"):
                # API call to get prediction score and threshold
                prediction_url = f"{api_url}/prediction/{st.session_state['selected_client']}"
                prediction_response = requests.get(prediction_url).json()
                prediction_score = prediction_response['score']
                threshold = prediction_response['threshold']

                prediction_percentage = round(prediction_score * 100, 2)

                st.subheader("Prediction Result")
                st.markdown(f"""
                    **Score Explanation:**
                    - This score of **{prediction_percentage}%** is an estimate of the probability that this client **will not repay** the credit (score = 1).
                    - If this score exceeds the threshold of **{round(threshold * 100, 2)}%**, the credit is **rejected**.
                    - Below this threshold, the credit is **approved**.
                """)

                score_text = "Credit Rejected" if prediction_score >= threshold else "Credit Approved"

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction_score * 100,
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
                            'value': threshold * 100
                        }
                    }
                ))

                st.plotly_chart(fig_gauge)
                st.markdown(f"**Decision:** {score_text} (Threshold: {round(threshold * 100, 2)}%)")

            # Button to display SHAP values
            if st.button("Show SHAP Analysis"):
                shap_values_url = f"{api_url}/shap_values/{st.session_state['selected_client']}"
                shap_response = requests.get(shap_values_url).json()

                if "shap_values" in shap_response:
                    # Convert to numpy array for SHAP
                    shap_values = shap_response['shap_values']
                    shap_values_np = np.array(shap_values)

                    # Check the dimension of SHAP values
                    if shap_values_np.ndim == 1:
                        shap_values_np = shap_values_np.reshape(1, -1)

                    # Replace `base_value` with a default value or received in the response if necessary
                    base_value = 0  # Use a fixed value if `explainer.expected_value` is not available

                    st.subheader("Expected Value")
                    st.markdown(f"The expected value of the model is: **{round(base_value, 4)}**.")

                    # Local visualization with waterfall plot
                    st.subheader("Local SHAP Values Analysis: Specific contribution of each feature for this client")
                    fig_local = plt.figure(figsize=(10, 6))
                    shap.waterfall_plot(shap.Explanation(values=shap_values_np[0], base_values=base_value, data=client_data.iloc[0]), show=False)
                    st.pyplot(fig_local)

                    # Global visualization with summary plot
                    st.subheader("Global SHAP Values Analysis: Importance of features across all clients")
                    fig_global = plt.figure(figsize=(10, 6))
                    shap.summary_plot(shap_values_np, client_data, plot_type="bar", show=False)
                    st.pyplot(fig_global)
                else:
                    st.error("Error retrieving SHAP values.")


    except Exception as e:
        st.error(f"Error retrieving client data: {str(e)}")
    st.subheader("© Streamlit dashboard by Laure Agrech")
