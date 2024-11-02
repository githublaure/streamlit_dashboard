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
api_url = "http://127.0.0.1:8000"

# Charger le SHAP explainer
explainer = joblib.load("../models/shap_explainer.pkl")

# Titre du dashboard
st.title("Tableau de bord de Prédiction du Crédit")

# Appel API pour obtenir les données des clients
data_clients_url = f"{api_url}/client_data/all"
try:
    response = requests.get(data_clients_url)
    response.raise_for_status()
    data_clients_response = response.json()
    client_ids = [client['SK_ID_CURR'] for client in data_clients_response['data']] if "data" in data_clients_response else []
except requests.exceptions.RequestException as e:
    st.error(f"Erreur lors du chargement des données clients : {str(e)}")
    client_ids = []

# Sélection de l'ID client via une `selectbox`
client_id = st.selectbox("Sélectionnez l'ID du client", client_ids)

# Conserver l'état du client sélectionné
if 'selected_client' not in st.session_state:
    st.session_state['selected_client'] = None

# Mettre à jour l'état du client lorsqu'un changement est effectué
if client_id:
    st.session_state['selected_client'] = client_id

# Vérifier si un client a été sélectionné
if st.session_state['selected_client']:
    # Appel API pour récupérer les données spécifiques du client
    client_data_url = f"{api_url}/client_data/{st.session_state['selected_client']}"
    try:
        client_response = requests.get(client_data_url).json()
        if "detail" in client_response:
            st.error(f"Erreur : {client_response['detail']}")
        else:
            client_data = pd.DataFrame(client_response['data'])

            # Afficher les données du client
            st.subheader("Données du client")
            st.write(client_data)

            # Extraction des features disponibles
            available_features = client_data.columns.tolist()

            # Sélection de la variable à comparer
            selected_feature = st.selectbox("Sélectionnez la variable à comparer", available_features)

            # Bouton pour afficher la comparaison de la variable
            if st.button("Afficher la comparaison de la variable"):
                st.subheader(f"Comparaison de la variable {selected_feature} entre le client et la moyenne")

                all_client_data_url = f"{api_url}/client_data/all_full"
                all_data_response = requests.get(all_client_data_url).json()
                if "data" in all_data_response:
                    all_data = pd.DataFrame(all_data_response['data'])
                    client_value = client_data[selected_feature].values[0]

                    if selected_feature not in all_data.columns:
                        st.error(f"La colonne {selected_feature} n'existe pas dans les données complètes.")
                    else:
                        mean_value = all_data[selected_feature].mean()

                        # Visualisation de la comparaison
                        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
                        sns.barplot(x=["client", "moyenne"], y=[client_value, mean_value], ax=ax[0])
                        ax[0].set_title(f"Comparaison du client à la moyenne ({selected_feature})")

                        sns.boxplot(x=all_data[selected_feature], ax=ax[1])
                        ax[1].set_title(f"Répartition de la variable {selected_feature} dans la clientèle")

                        st.pyplot(fig)

            # Bouton pour afficher le score et la jauge
            if st.button("Afficher le score de prédiction"):
                # Appel API pour obtenir la prédiction et le threshold
                prediction_url = f"{api_url}/prediction/{st.session_state['selected_client']}"
                prediction_response = requests.get(prediction_url).json()
                prediction_score = prediction_response['score']
                threshold = prediction_response['threshold']

                prediction_percentage = round(prediction_score * 100, 2)

                st.subheader("Résultat de la prédiction")
                st.markdown(f"""
                    **Explication du score :**
                    - Ce score de **{prediction_percentage}%** est une estimation de la probabilité que ce client **ne rembourse pas** le crédit (score = 1).
                    - Si ce score dépasse le seuil de **{round(threshold * 100, 2)}%**, le crédit est **refusé**.
                    - En dessous de ce seuil, le crédit est **accordé**.
                """)

                score_text = "Crédit refusé" if prediction_score >= threshold else "Crédit accordé"

                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prediction_score * 100,
                    title={'text': f"Score de Prédiction: {prediction_percentage}%"},
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
                st.markdown(f"**Décision :** {score_text} (Seuil: {round(threshold * 100, 2)}%)")

            # Bouton pour afficher les valeurs SHAP
            if st.button("Afficher l'analyse SHAP"):
                shap_values_url = f"{api_url}/shap_values/{st.session_state['selected_client']}"
                shap_response = requests.get(shap_values_url).json()

                if "shap_values" in shap_response:
                    shap_values = shap_response['shap_values']
                    shap_values_np = np.array(shap_values[0])

                    if np.ndim(explainer.expected_value) > 0:
                        base_value = float(explainer.expected_value[0])  # Extraire le premier élément si c'est un tableau
                    else:
                        base_value = float(explainer.expected_value)  # Utiliser directement si c'est un scalaire


                    st.subheader("Valeur attendue (Expected Value)")
                    st.markdown(f"La valeur attendue du modèle est : **{round(base_value, 4)}**.")

                    st.subheader("Analyse locale des SHAP values : Contribution spécifique de chaque feature pour ce client")
                    fig_local = plt.figure(figsize=(10, 6))
                    shap.waterfall_plot(shap.Explanation(values=shap_values_np, base_values=base_value, data=client_data.iloc[0, :]), show=False)
                    st.pyplot(fig_local)

                    st.subheader("Analyse globale des SHAP values : Importance des features à travers tous les clients")
                    shap_values_matrix = shap_values_np.reshape(1, -1) if shap_values_np.ndim == 1 else shap_values_np
                    fig_global = plt.figure(figsize=(10, 6))
                    shap.summary_plot(shap_values_matrix, client_data, plot_type="bar", show=False)
                    st.pyplot(fig_global)
                else:
                    st.error("Erreur lors de la récupération des valeurs SHAP.")
    except Exception as e:
        st.error(f"Erreur lors de la récupération des données du client : {str(e)}")
