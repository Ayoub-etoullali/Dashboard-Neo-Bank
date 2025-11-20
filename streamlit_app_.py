import streamlit as st
import pandas as pd
import requests
import os
import logging
import re # Pour les expressions régulières afin de parser les explications SHAP
from streamlit_echarts import st_echarts

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="Dashboard Néo-Banque", layout="centered")

# Modern fintech color palette
PALETTE = {
    "primary": "#0B63D6",       # Blue main
    "positive_bg": "#e6f4ea",   # Light green
    "positive_text": "#2d8a4a", # Dark green
    "negative_bg": "#fdeaea",   # Light red
    "negative_text": "#d93025", # Dark red
    "warning_bg": "#fff8e6",    # Light yellow
    "warning_text": "#fbbc04",  # Orange
    "info_bg": "#f5f5f5",       # Neutral gray
    "shadow": "0 4px 12px rgba(0,0,0,0.08)"
}

# Typography
FONT_SIZE_TITLE = "28px"
FONT_SIZE_SUBTITLE = "20px"
FONT_SIZE_TEXT = "16px"


# Fonction pour initialiser ou réinitialiser l'état de la session
def reset_scoring_state():
    """Réinitialise les variables de session liées au scoring et aux explications SHAP."""
    if 'score' in st.session_state:
        del st.session_state['score']
    if 'explanations' in st.session_state:
        del st.session_state['explanations']
    st.session_state['api_called'] = False

# Fonction utilitaire pour parser les explications SHAP selon le nouveau format
def _parse_shap_explanation(explanation_text):
    """
    Parse une chaîne d'explication SHAP selon le nouveau format.
    Ex: "La fonctionnalité 'revenu' impacte positivement la prédiction avec un poids de 0.228"
    Ex: "La feature 'age' impacte négativement la prédiction avec un poids de -0.228"
    Retourne (direction_symbole_html, description, valeur_str, est_positif)
    """
    match_positive = re.match(r"La (fonctionnalité|feature) '(.+?)' impacte positivement la prédiction avec un poids de (\d+\.?\d*)", explanation_text)
    match_negative = re.match(r"La (fonctionnalité|feature) '(.+?)' impacte négativement la prédiction avec un poids de (-\d+\.?\d*)", explanation_text)

    # Flèche à angle droit : HTML unicode character
    arrow_html = "&#10148;" # Pour une flèche orientée à droite

    if match_positive:
        feature_name = match_positive.group(2).strip()
        value = match_positive.group(3)
        description = f"La fonctionnalité '{feature_name}' impacte positivement la prédiction avec un poids de "
        return arrow_html, description, value, True
    elif match_negative:
        feature_name = match_negative.group(2).strip()
        value = match_negative.group(3)
        description = f"La fonctionnalité '{feature_name}' impacte négativement la prédiction avec un poids de "
        return arrow_html, description, value, False
    else:
        # Cas par défaut si le format ne correspond pas
        return arrow_html, explanation_text, "", False # Par défaut, mettons-le en rouge pour signaler un problème de parsing


# Fonction pour afficher un facteur SHAP stylisé
def _display_shap_factor(direction_symbol_html, description, value, is_positive):
    """Affiche un facteur SHAP avec le style conditionnel."""
    # Couleurs de fond et de texte basées sur l'image fournie
    bg_color = "#e6ffe6" if is_positive else "#ffe6e6" # Vert clair si positif, rouge clair si négatif
    score_color = "#269f67" if is_positive else "#ea4521" # Vert foncé si positif, rouge foncé si négatif
    arrow_color = "#6c757d" # Couleur de la flèche (gris) comme sur l'image

    # Style pour la boîte globale
    container_style = f"""
        padding: 10px;
        margin-bottom: 8px;
        border-radius: 8px;
        background-color: {bg_color};
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); /* Ombre légère */
        min-height: 40px; /* Assure une hauteur minimale */
    """
    # Style pour la description et la flèche
    desc_style = f"""
        font-size: 16px;
        color: #333; /* Couleur du texte de la description */
        font-weight: 500;
        display: flex;
        align-items: center;
        flex-grow: 1; /* Permet à la description de prendre l'espace disponible */
    """
    # Style pour la valeur (score)
    value_style = f"""
        font-size: 16px;
        font-weight: bold;
        color: {score_color};
        text-align: right;
        min-width: 60px; /* Assure un espace suffisant pour la valeur */
    """
    # Style pour la flèche
    arrow_style = f"""
        margin-right: 10px;
        font-size: 20px;
        color: {arrow_color}; /* La flèche est toujours grise selon l'image */
    """

    html_content = f"""
    <div style="{container_style}">
        <div style="{desc_style}">
            <span style="{arrow_style}">{direction_symbol_html}</span>
            <span>{description}</span>
        </div>
        <div style="{value_style}">
            {value}
        </div>
    </div>
    """
    st.markdown(html_content, unsafe_allow_html=True)


# Ajout de la section RGPD dans la sidebar
with st.sidebar.expander("🔐 Données & RGPD"):
    st.markdown("""
    **Conformité RGPD**

    - Ce dashboard traite des données **pseudonymisées**.
    - Aucune donnée personnelle (nom, email...) n’est utilisée.
    - Les données sont utilisées uniquement à des fins de **scoring de prêt**.
    - Le traitement est **explicable** grâce aux outils SHAP.

    👉 Ce traitement respecte les principes du RGPD :
    - Licéité, transparence, finalité, minimisation.
    - Pas de stockage ni de profilage automatisé externe.
    """)

# Charger les données des clients
try:
    clients = pd.read_csv("data/clients.csv").reset_index().rename(columns={"index": "id"})
except FileNotFoundError:
    st.error("Erreur : Le fichier 'data/clients.csv' est introuvable. Veuillez le placer dans le répertoire 'data'.")
    st.stop()

st.title("📊 Dashboard conseiller")

# Sélection du client
selected_id = st.selectbox(
    "Choisir un client",
    clients.index,
    key="client_selector",
    on_change=reset_scoring_state # Appelle la fonction de réinitialisation
)

client = clients.loc[selected_id]

# Initialisation de api_called à False si ce n'est pas déjà fait
if 'api_called' not in st.session_state:
    st.session_state['api_called'] = False

# Utilisation de st.columns pour un agencement comme sur l'image
col1, col2 = st.columns([0.6, 0.4])

with col1:
    st.subheader("Informations Client")

    # Création d'un DataFrame pour les informations du client
    client_info_df = pd.DataFrame({
        "Caractéristique": [
            "ID Client",
            "Âge",
            "Revenu annuel",
            "Ancienneté",
            "Incidents de paiement",
            "Score de crédit initial"
        ],
        "Valeur": [
            str(client['id']),
            f"{int(client['age'])} ans",
            f"{float(client['revenu']):,.0f} €",
            f"{int(client['anciennete'])} ans",
            str(int(client['nb_incidents'])),
            f"{float(client['score_credit']):.1f}"
        ]
    })

    # Afficher le DataFrame
    st.dataframe(client_info_df, hide_index=True)

    # Bouton pour envoyer pour scoring
    if st.button("📤 Envoyer pour scoring", key="score_button"):
        API_URL = os.getenv("API_URL", "https://dashboard-neo-bank.onrender.com/predict")

        input_data = {
            "age": int(client["age"]),
            "revenu": float(client["revenu"]),
            "anciennete": int(client["anciennete"]),
            "nb_incidents": int(client["nb_incidents"]),
            "score_credit": float(client["score_credit"]),
        }

        try:
            logging.info(f"Envoi de la requête à l'API pour le client ID: {client['id']}")
            res = requests.post(API_URL, json=input_data, timeout=10)
            res.raise_for_status()
            response_data = res.json()
            score = response_data["score"]
            explanations_from_api = response_data.get("explanations", [])

            # Stockage des résultats dans st.session_state
            st.session_state['score'] = score
            st.session_state['explanations'] = explanations_from_api
            st.session_state['api_called'] = True

        except requests.exceptions.ConnectionError:
            st.error(f"Erreur de connexion à l’API. Vérifiez que l'API est accessible à l'adresse {API_URL}.")
            reset_scoring_state()
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur lors de la communication avec l’API : {e}. Vérifiez l'URL et la configuration de l'API.")
            reset_scoring_state()
        except Exception as e:
            st.error(f"Une erreur inattendue s'est produite : {e}")
            reset_scoring_state()

# Vérifier si l'API a été appelée et afficher les résultats (jauge et SHAP)
if st.session_state['api_called']:
    score = st.session_state['score']
    explanations_from_api = st.session_state['explanations']

    with col2:
        st.subheader("Score de Crédit")
        credit_score_percentage = round(score * 100, 1)

        options = {
            "series": [
                {
                    "type": "gauge",
                    "axisLine": {
                        "lineStyle": {
                            "width": 10,
                            "color": [
                                [0.5, "#ea4521"],  # Rouge pour <= 50%
                                [0.8, "#f7bb10"],  # Jaune pour <= 80%
                                [1, "#269f67"]     # Vert pour > 80%
                            ]
                        }
                    },
                    "pointer": {"show": False},
                    "axisTick": {"show": False},
                    "splitLine": {"show": False},
                    "axisLabel": {"show": False},
                    "detail": {
                        "show": True,
                        "offsetCenter": [0, "-10%"],
                        "valueAnimation": True,
                        "formatter": "{value}%",
                        "fontSize": 30,
                        "fontWeight": "bolder",
                        "color": "#333",
                    },
                    "title": {
                        "show": True,
                        "offsetCenter": [0, "120%"],
                        "fontSize": 14,
                        "color": "#333",
                        "formatter": ""
                    },
                    "data": [{"value": credit_score_percentage}],
                    "progress": {
                        "show": True,
                        "width": 10
                    },
                    "splitNumber": 0,
                    "radius": "80%",
                    "center": ["50%", "50%"],
                    "min": 0,
                    "max": 100,
                    "anchor": {"show": False},
                    "itemStyle": {"color": "#269f67"},
                }
            ]
        }
        st_echarts(options=options, height="200px")

        if score > 0.8:
            st.markdown("<p style='text-align: center; color: #269f67; font-weight: bold;'>✅ Éligible</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Score excellent pour l'octroi de crédit</p>", unsafe_allow_html=True)
        elif score > 0.5:
            st.markdown("<p style='text-align: center; color: #f7bb10; font-weight: bold;'>⚠️ Potentiellement éligible</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Score bon pour l'octroi de crédit, mais à étudier</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='text-align: center; color: #ea4521; font-weight: bold;'>❌ Inéligible probable</p>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center;'>Score faible pour l'octroi de crédit</p>", unsafe_allow_html=True)

    st.subheader("Comprendre le score (Facteurs d'influence SHAP)")
    if explanations_from_api:
        # Appliquer le style conditionnel aux explications SHAP
        for explanation_text in explanations_from_api:
            direction_symbol_html, description, value, is_positive = _parse_shap_explanation(explanation_text)
            _display_shap_factor(direction_symbol_html, description, value, is_positive)
    else:
        st.info("Aucune explication détaillée disponible pour le moment ou une erreur s'est produite côté API.")
else:
    with col2:
        st.subheader("Score de Crédit")
        st.info("Cliquez sur 'Envoyer pour scoring' pour calculer le score.")
    st.subheader("Comprendre le score (Facteurs d'influence SHAP)")
    st.info("Les facteurs d'influence SHAP apparaîtront après le calcul du score.")
