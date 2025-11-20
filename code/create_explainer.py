import joblib
import shap
import sys
import os
import pandas as pd # <-- Assurez-vous d'avoir pandas importé

def main():
    if not os.path.exists("model"):
        os.makedirs("model")

    try:
        print("Chargement du modèle depuis model/model.pkl...")
        model = joblib.load("model/model.pkl")

        # --- AJOUT OU VÉRIFICATION DE CETTE PARTIE ---
        print("Chargement des données de référence pour SHAP depuis data/clients.csv...")
        # Utilisez un échantillon représentatif de vos données d'entraînement comme background dataset
        # Un échantillon de 100 à 1000 lignes est généralement suffisant.
        shap_data_reference = pd.read_csv("data/clients.csv").sample(n=min(len(pd.read_csv("data/clients.csv")), 100), random_state=42)
        # J'ajoute min(len(...), 100) pour éviter une erreur si clients.csv a moins de 100 lignes
        # et pour prendre 100 lignes si clients.csv est grand.
        # ---------------------------------------------

    except FileNotFoundError:
        print("Erreur: fichier model/model.pkl ou data/clients.csv introuvable")
        print("Veuillez d'abord entraîner et sauvegarder le modèle et les données")
        sys.exit(1)
    except Exception as e:
        print(f"Erreur lors du chargement du modèle ou des données: {str(e)}")
        sys.exit(1)

    try:
        print("Création de l'explainer SHAP avec background dataset...")
        explainer = shap.TreeExplainer(model, shap_data_reference) # <-- PASSEZ shap_data_reference ICI

        print("Sauvegarde de l'explainer...")
        joblib.dump(explainer, "model/explainer.pkl")
        print("Succès: Explainer sauvegardé dans model/explainer.pkl")
    except Exception as e:
        print(f"Erreur lors de la création de l'explainer: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()