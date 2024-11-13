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
api_url = "http://54.198.38.13/"

# Charger le SHAP explainer
explainer = joblib.load("../models/shap_explainer.pkl")

# Titre du dashboard
st.title("Tableau de bord de Prédiction du Crédit")

# Appel API pour obtenir les données des clients complets (par exemple, les IDs clients)
data_clients_url = f"{api_url}/client_data/all"
try:
    response = requests.get(data_clients_url)
    response.raise_for_status()  # Vérifie les erreurs HTTP
    data_clients_response = response.json()
    
    # Assurez-vous que la clé "data" est bien présente dans la réponse
    if "data" not in data_clients_response:
        st.error("Erreur : La réponse de l'API ne contient pas de 'data'.")
        client_ids = []
    else:
        client_ids = [client['SK_ID_CURR'] for client in data_clients_response['data']]
except requests.exceptions.RequestException as e:
    st.error(f"Erreur lors du chargement des données clients : {str(e)}")
    client_ids = []

# Sélection de l'ID client via une `selectbox`
client_id = st.selectbox("Sélectionnez l'ID du client", client_ids)

# Sauvegarder l'ID client sélectionné dans `session_state`
if 'client_id' not in st.session_state:
    st.session_state['client_id'] = client_id

# Appel API pour récupérer les données complètes des clients (avec toutes les colonnes)
all_client_data_url = f"{api_url}/client_data/all_full"
try:
    all_data_response = requests.get(all_client_data_url)
    all_data_response.raise_for_status()  # Vérifie les erreurs HTTP
    all_data_response = all_data_response.json()
    if "data" not in all_data_response:
        st.error(f"Erreur : Les données complètes des clients ne sont pas disponibles.")
        all_data = pd.DataFrame()
    else:
        all_data = pd.DataFrame(all_data_response['data'])  # Toutes les données clients complètes
except requests.exceptions.RequestException as e:
    st.error(f"Erreur lors du chargement des données complètes : {str(e)}")
    all_data = pd.DataFrame()  # Crée un dataframe vide par défaut

# Vérifier si des données clients sont disponibles
if not all_data.empty:
    # Appel API pour récupérer les données spécifiques du client sélectionné
    client_data_url = f"{api_url}/client_data/{client_id}"
    client_response = requests.get(client_data_url).json()

    if "detail" in client_response:
        st.error(f"Erreur : {client_response['detail']}")
    else:
        client_data = pd.DataFrame(client_response['data'])

        # Afficher les données du client
        st.subheader("Données du client")
        st.write(client_data)

        # Extraire dynamiquement la liste des features disponibles dans le dataset du client
        available_features = client_data.columns.tolist()

        # Sélection de la feature à comparer
        selected_feature = st.selectbox("Sélectionnez la variable à comparer", available_features)

        # Comparaison de la variable sélectionnée entre le client et la moyenne de la population
        st.subheader(f"Comparaison de la variable {selected_feature} entre le client et la moyenne")

        # Comparer la variable entre le client et la moyenne de la population
        client_value = client_data[selected_feature].values[0]

        # Vérifier si la variable existe dans toutes les données
        if selected_feature not in all_data.columns:
            st.error(f"La colonne {selected_feature} n'existe pas dans les données complètes.")
        else:
            mean_value = all_data[selected_feature].mean()

            # Visualisation de la comparaison
            fig, ax = plt.subplots(1, 2, figsize=(12, 4))

            # Comparaison client vs moyenne
            sns.barplot(x=["client", "moyenne"], y=[client_value, mean_value], ax=ax[0])
            ax[0].set_title(f"Comparaison du client à la moyenne ({selected_feature})")

            # Distribution dans la population
            sns.boxplot(x=all_data[selected_feature], ax=ax[1])
            ax[1].set_title(f"Répartition de la variable {selected_feature} dans la clientèle")

            st.pyplot(fig)  # Affichage du graphique

        # Appel API pour obtenir la prédiction et le threshold
        prediction_url = f"{api_url}/prediction/{client_id}"
        prediction_response = requests.get(prediction_url).json()
        prediction_score = prediction_response['score']
        threshold = prediction_response['threshold']  # Le threshold récupéré via l'API

        # Calcul du score business basé sur le F-beta
        prediction_percentage = round(prediction_score * 100, 2)
        
        # Affichage du résultat sous forme d'indicateur
        st.subheader("Résultat de la prédiction")
        st.markdown(f"""
            **Explication du score :**
            - Ce score de **{prediction_percentage}%** est une estimation de la probabilité que ce client **ne rembourse pas** le crédit (score = 1).
            - Si ce score dépasse le seuil de **{round(threshold * 100, 2)}%**, le crédit est **refusé**.
            - En dessous de ce seuil, le crédit est **accordé**.
        """)

        score_text = "Crédit refusé" if prediction_score >= threshold else "Crédit accordé"

        # Gauge chart (Jauge de score) avec le score de prédiction et le threshold
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction_score * 100,  # Score en pourcentage
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
                    'value': threshold * 100  # Ligne de threshold
                }
            }
        ))

        st.plotly_chart(fig_gauge)

        # Afficher la décision finale
        st.markdown(f"**Décision :** {score_text} (Seuil: {round(threshold * 100, 2)}%)")

        # Appel API pour obtenir les SHAP values locales
        shap_values_url = f"{api_url}/shap_values/{client_id}"
        shap_response = requests.get(shap_values_url).json()
        shap_values = shap_response['shap_values']

        # Correction pour éviter l'erreur avec les listes
        shap_values_np = np.array(shap_values[0])  # Convertir en tableau NumPy

        # Assurer que `base_values` est un seul float (pour les modèles binaires ou multi-classes)
        if isinstance(explainer.expected_value, np.ndarray) and len(explainer.expected_value) > 1:
            base_value = explainer.expected_value[0]  # Pour un modèle multi-classe
        else:
            base_value = float(explainer.expected_value)  # Modèle binaire

        # Affichage de la valeur attendue (expected value)
        st.subheader("Valeur attendue (Expected Value)")
        st.markdown(f"La valeur attendue du modèle est : **{round(base_value, 4)}**.")
        st.markdown("C'est la prédiction moyenne du modèle avant de prendre en compte les contributions spécifiques de chaque variable.")

        # **Waterfall plot** : Analyse locale des SHAP values
        st.subheader("Analyse locale des SHAP values : Contribution spécifique de chaque feature pour ce client")
        st.markdown("Ce graphique montre comment les différentes features influencent la prédiction pour ce client particulier. Chaque barre indique l'impact d'une feature qui pousse la prédiction soit vers le haut, soit vers le bas.")
        fig_local = plt.figure(figsize=(10, 6))
        shap.waterfall_plot(shap.Explanation(values=shap_values_np, base_values=base_value, data=client_data.iloc[0, :]), show=False)
        st.pyplot(fig_local)  # Affichage du graphique Waterfall avec st.pyplot

        # **Bar plot** : Analyse globale des SHAP values
        st.subheader("Analyse globale des SHAP values : Importance des features à travers tous les clients")
        st.markdown("Ce graphique montre quelles features sont globalement les plus importantes pour le modèle, en moyenne, à travers tous les clients.")
        fig_global = plt.figure(figsize=(10, 6))
        shap.summary_plot(np.array(shap_values), client_data, plot_type="bar", show=False)
        st.pyplot(fig_global)  # Affichage du graphique bar avec st.pyplot
