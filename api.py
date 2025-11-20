from fastapi import FastAPI, HTTPException # Importe HTTPException pour des réponses d'erreur claires
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os # Pour vérifier l'existence des fichiers

app = FastAPI(
    title="API de Prédiction de Risque Client Neo-Banque",
    description="API pour prédire le risque client et fournir des explications SHAP."
)

# 1. Définition des données attendues pour la requête POST /predict
class ClientData(BaseModel):
    age: int
    revenu: float
    anciennete: int # Ancienneté du client en mois ou années
    nb_incidents: int # Nombre d'incidents (par exemple, de paiement ou techniques)
    score_credit: int # Score de crédit actuel du client (ex: FICO, score interne)

# 2. Définition des chemins absolus ou relatifs vers les fichiers du modèle et de l'explainer
# Assurez-vous que ces chemins sont corrects sur votre système et dans l'environnement de déploiement (Render).
MODEL_PATH = "model/model.pkl"
EXPLAINER_PATH = "model/explainer.pkl"

# Initialisation des variables globales pour le modèle et l'explainer
model = None
explainer = None

# Ordre attendu des fonctionnalités - C'EST CRUCIAL !
# CETTE LISTE DOIT CORRESPONDRE EXACTEMENT À L'ORDRE ET AUX NOMS DES COLONNES
# QUI ONT ÉTÉ UTILISÉES POUR ENTRAÎNER VOTRE MODÈLE ET CRÉER VOTRE EXPLAINER SHAP.
expected_feature_order = ['age', 'revenu', 'anciennete', 'nb_incidents', 'score_credit']


# 3. Chargement du modèle et de l'explainer au démarrage de l'application (une seule fois)
@app.on_event("startup")
def load_resources():
    global model, explainer # Permet de modifier les variables globales définies plus haut

    print("Tentative de chargement des ressources (modèle et explainer)...")

    # Vérification de l'existence des fichiers pour un démarrage robuste
    if not os.path.exists(MODEL_PATH):
        # Utilisation de RuntimeError pour empêcher le démarrage de l'API si les fichiers essentiels manquent
        raise RuntimeError(f"Fichier modèle non trouvé à : {MODEL_PATH}. L'application ne peut pas démarrer.")
    if not os.path.exists(EXPLAINER_PATH):
        raise RuntimeError(f"Fichier explainer non trouvé à : {EXPLAINER_PATH}. L'application ne peut pas démarrer.")

    # Tentative de chargement des objets
    try:
        model = joblib.load(MODEL_PATH)
        explainer = joblib.load(EXPLAINER_PATH)
        print("✅ Modèle et Explainer chargés avec succès.")
    except Exception as e:
        # En cas d'erreur de chargement (ex: fichier corrompu, problème de compatibilité), lever une exception
        raise RuntimeError(f"❌ Erreur lors du chargement du modèle ou de l'explainer : {e}. L'application ne peut pas démarrer.")

# 4. Point de terminaison API pour la prédiction de risque et l'explication SHAP
@app.post("/predict")
def predict(data: ClientData):
    print(f"✅ Données reçues pour prédiction : {data.dict()}")

    # Convertir les données Pydantic en DataFrame Pandas
    df = pd.DataFrame([data.dict()])

    # --- Étape CRUCIALE : Réindexer le DataFrame pour assurer l'ordre correct des colonnes ---
    # Cela garantit que les fonctionnalités sont passées à l'explainer et au modèle
    # dans le même ordre et avec les mêmes noms que celui utilisé pendant l'entraînement.
    try:
        df = df[expected_feature_order]
        print(f"✅ DataFrame réindexé avec l'ordre attendu : {df.columns.tolist()}")
    except KeyError as e:
        # Si une fonctionnalité attendue (définie dans expected_feature_order) est manquante dans les données d'entrée
        error_msg = (f"❌ Erreur de clé lors de la réindexation du DataFrame : {e}. "
                     f"Assurez-vous que toutes les fonctionnalités de `expected_feature_order` sont présentes dans les données reçues.")
        print(error_msg)
        # Retourne une erreur HTTP 400 (Bad Request) au client
        raise HTTPException(status_code=400, detail=error_msg)

    # --- DÉBOGAGE PRÉ-PRÉDICTION : Affichage du DataFrame avant son utilisation ---
    print("\n--- DÉBOGAGE PRÉ-PRÉDICTION ---")
    print("DataFrame final à utiliser pour la prédiction et SHAP :")
    print(df)
    print(f"Forme du DataFrame (df.shape) : {df.shape}")
    print(f"Colonnes du DataFrame (df.columns) : {df.columns.tolist()}")
    print("--- FIN DÉBOGAGE PRÉ-PRÉDICTION ---\n")

    # 5. Prédiction du score (probabilité de la classe positive)
    score = None # Initialise le score à None pour s'assurer qu'il est défini ou qu'une erreur est levée
    try:
        # model.predict_proba renvoie généralement un tableau numpy de forme (n_samples, n_classes)
        # Pour une classification binaire, c'est (n_samples, 2).
        # [0][1] prend la probabilité du premier échantillon pour la classe positive (index 1).
        prediction_proba = model.predict_proba(df)
        print(f"DEBUG: Type de sortie de model.predict_proba : {type(prediction_proba)}")
        print(f"DEBUG: Forme de sortie de model.predict_proba : {prediction_proba.shape}")
        
        # Vérification de la forme pour extraire correctement la probabilité de la classe positive
        if prediction_proba.ndim == 2 and prediction_proba.shape[1] > 1:
            score = prediction_proba[0][1] # Probabilité pour le premier échantillon, classe 1
        elif prediction_proba.ndim == 1:
            # Cas où predict_proba renverrait directement une probabilité 1D (moins commun pour binaire)
            score = prediction_proba[0]
        else:
            raise ValueError(f"Forme inattendue du résultat de predict_proba: {prediction_proba.shape}. Attendu (1, 2) ou (1,).")
            
        print(f"✅ Score de prédiction calculé : {score:.4f}")

    except Exception as e:
        # En cas d'erreur lors de la prédiction, retourner une erreur HTTP 500
        error_msg = f"❌ Erreur lors du calcul du score de prédiction : {e}. Vérifiez le modèle ou les données d'entrée."
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

    # 6. Calcul des valeurs SHAP pour l'explicabilité du modèle
    raw_shap_output = None # Initialise raw_shap_output à None
    try:
        raw_shap_output = explainer.shap_values(df)
    except Exception as e:
        # En cas d'erreur lors du calcul SHAP, retourner une erreur HTTP 500
        error_msg = f"❌ Erreur lors du calcul des valeurs SHAP : {e}. L'explainer est-il correct et compatible avec le modèle et les données ?"
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

    # --- DÉBOGAGE DE LA SORTIE SHAP BRUTE (raw_shap_output) : TRÈS UTILE POUR COMPRENDRE LA STRUCTURE ---
    print(f"\n--- DÉBOGAGE DE LA SORTIE SHAP BRUTE (raw_shap_output) ---")
    print(f"Type de raw_shap_output : {type(raw_shap_output)}")
    if isinstance(raw_shap_output, list):
        print(f"raw_shap_output est une liste. Longueur : {len(raw_shap_output)}")
        for i, item in enumerate(raw_shap_output):
            if isinstance(item, np.ndarray):
                print(f"  Élément {i} (ndarray) : Forme : {item.shape}, Dimensions : {item.ndim}")
            else:
                print(f"  Élément {i} : Type : {type(item)}")
    elif isinstance(raw_shap_output, np.ndarray):
        print(f"raw_shap_output est un ndarray. Forme : {raw_shap_output.shape}, Dimensions : {raw_shap_output.ndim}")
    else:
        print(f"Type de raw_shap_output inattendu : {type(raw_shap_output)}")
    print(f"--- FIN DU DÉBOGAGE DE LA SORTIE SHAP BRUTE ---\n")

    # 7. Extraction robuste des shap_values pour la classe positive (LA CORRECTION CLÉ EST ICI)
    shap_values = None
    if isinstance(raw_shap_output, list):
        # C'est le cas typique si TreeExplainer retourne une liste de ndarrays pour chaque classe
        if len(raw_shap_output) == 2 and isinstance(raw_shap_output[1], np.ndarray):
            # raw_shap_output[1] contient les SHAP pour la classe positive.
            # Sa forme peut être (n_samples, n_features) ou (n_features,) si n_samples=1
            if raw_shap_output[1].ndim == 2 and raw_shap_output[1].shape[0] > 0:
                shap_values = raw_shap_output[1][0] # Premier échantillon
            elif raw_shap_output[1].ndim == 1:
                shap_values = raw_shap_output[1] # Déjà 1D pour toutes les features
            else:
                print(f"Avertissement : raw_shap_output[1] a un nombre de dimensions inattendu : {raw_shap_output[1].ndim}.")
        # Fallback pour d'autres structures de liste (moins probable pour votre cas)
        elif len(raw_shap_output) > 0 and isinstance(raw_shap_output[0], np.ndarray):
            if raw_shap_output[0].ndim == 2 and raw_shap_output[0].shape[0] > 0:
                shap_values = raw_shap_output[0][0]
            elif raw_shap_output[0].ndim == 1:
                shap_values = raw_shap_output[0]
            else:
                print(f"Avertissement : raw_shap_output[0] a un nombre de dimensions inattendu : {raw_shap_output[0].ndim}.")
    elif isinstance(raw_shap_output, np.ndarray):
        # C'EST LE CAS QUE VOS LOGS ONT MONTRÉ : Forme (1, 5, 2)
        if raw_shap_output.ndim == 3: # Forme (n_samples, n_features, n_classes)
            # Pour extraire les SHAP du premier échantillon (index 0), pour toutes les fonctionnalités (slice :),
            # et pour la classe positive (index 1 de la dernière dimension, qui est les classes)
            if raw_shap_output.shape[0] > 0 and raw_shap_output.shape[2] > 1:
                shap_values = raw_shap_output[0, :, 1] # <<<<< C'EST LA LIGNE CLÉ CORRIGÉE
            else:
                print(f"Avertissement : raw_shap_output 3D n'a pas la forme attendue pour l'extraction de la classe positive.")
        elif raw_shap_output.ndim == 2: # Forme (n_samples, n_features)
            if raw_shap_output.shape[0] > 0:
                shap_values = raw_shap_output[0, :] # Prendre le premier échantillon
            else:
                print(f"Avertissement : raw_shap_output 2D est vide.")
        elif raw_shap_output.ndim == 1: # Forme (n_features,)
            shap_values = raw_shap_output
    
    # Vérification finale : Si shap_values n'a toujours pas été extrait, lever une erreur
    if shap_values is None:
        error_msg = "❌ Impossible d'extraire les valeurs SHAP de la structure renvoyée par l'explainer. La structure de raw_shap_output est inattendue ou non gérée."
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)
    
    # 8. Vérification de la cohérence entre le nombre de noms de features et les valeurs SHAP extraites
    feature_names = df.columns.tolist() # Utilise les noms de colonnes du DataFrame réindexé (garantit l'ordre correct)
    print(f"feature_names (issues du DataFrame réindexé) : {feature_names}")
    print(f"shap_values.shape (extraites après correction) : {shap_values.shape}")
    
    if len(feature_names) != len(shap_values):
        # Si cette erreur se produit encore après la correction, cela signifie que
        # l'explainer lui-même a été créé de manière incorrecte et ne produit pas
        # les SHAP pour toutes les features attendues.
        error_msg = (f"❌ ERREUR MAJEURE PERSISTANTE : Le nombre de noms de fonctionnalités ({len(feature_names)}) "
                     f"ne correspond pas au nombre de valeurs SHAP ({len(shap_values)}). "
                     f"Cela indique une INCOHÉRENCE fondamentale entre l'explainer SHAP et les données d'entraînement. "
                     f"Veuillez VÉRIFIER IMPÉRATIVEMENT comment votre `explainer.pkl` a été créé dans `create_explainer.py` "
                     f"et assurez-vous qu'il est entraîné sur TOUTES les fonctionnalités dans le BON ORDRE.")
        print(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

    # 9. Création du dictionnaire des impacts SHAP {feature_name: shap_value}
    shap_impacts = {}
    for i, feature in enumerate(feature_names):
        shap_impacts[feature] = float(shap_values[i]) # Convertir en float pour une sérialisation JSON sûre

    # 10. Tri des impacts par valeur absolue décroissante pour l'affichage
    sorted_impacts = sorted(shap_impacts.items(), key=lambda x: abs(x[1]), reverse=True)

    # 11. Construction de la liste d'explications lisibles par l'utilisateur
    explanation_list = []
    for feature, impact in sorted_impacts:
        sign = "positivement" if impact > 0 else "négativement"
        explanation_list.append(f"La fonctionnalité '{feature}' impacte {sign} la prédiction avec un poids de {impact:.3f}")
    
    print("✅ Prédiction et explications SHAP générées avec succès.")

    # 12. Retour de la réponse finale de l'API
    return {
        "score": score, # Le score de risque calculé
        "shap_values": shap_impacts, # Les valeurs SHAP brutes par fonctionnalité
        "explanations": explanation_list # Les explications lisibles pour le dashboard
    }
