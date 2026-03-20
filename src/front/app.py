"""Interface Streamlit pour la prédiction de fleurs Iris via une API FastAPI.

Ce module fournit une interface utilisateur développée avec Streamlit
permettant de saisir les caractéristiques d'une fleur Iris, d'envoyer
ces données à une API de prédiction, puis d'afficher l'espèce prédite
ainsi que la version active du modèle enregistrée dans MLflow.

Fonctionnalités principales :
- Saisie manuelle des features de l'Iris
- Chargement d'une ligne depuis un dataset de test CSV
- Appel HTTP à l'API `/predict`
- Affichage de la prédiction et de la version du modèle en production
"""

import os
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Ajoute la racine du projet au PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Iris Prediction App",
    page_icon="🌸",
    layout="wide",
)

# -----------------------------
# CONSTANTES
# -----------------------------
CLASS_NAMES = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica",
}

TEST_DATA_PATH = ROOT_DIR / "data" / "iris_test.csv"


# -----------------------------
# CHARGEMENT DATASET DE TEST
# -----------------------------
@st.cache_data
def load_test_dataset() -> pd.DataFrame | None:
    """Charge le dataset de test Iris depuis un fichier CSV.

    Cette fonction lit le fichier `data/iris_test.csv` si celui-ci existe
    dans le projet. Le résultat est mis en cache avec `st.cache_data`
    afin d'éviter de relire le fichier à chaque rerun de l'application.

    Returns:
        pd.DataFrame | None:
            Le dataset de test sous forme de DataFrame pandas si le fichier
            existe, sinon `None`.
    """
    if TEST_DATA_PATH.exists():
        return pd.read_csv(TEST_DATA_PATH)
    return None


# -----------------------------
# INITIALISATION SESSION STATE
# -----------------------------
def init_session_state() -> None:
    """Initialise les valeurs par défaut dans `st.session_state`.

    Cette fonction définit des valeurs initiales pour les sliders du formulaire
    ainsi que pour l'index de la ligne sélectionnée dans le dataset de test.
    Les valeurs ne sont ajoutées que si elles n'existent pas déjà dans
    `st.session_state`.

    Returns:
        None
    """
    defaults = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
        "selected_test_index": 0,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# -----------------------------
# CHARGER UNE LIGNE DE TEST
# -----------------------------
def load_test_row_into_form(df: pd.DataFrame, row_index: int) -> None:
    """Charge une ligne du dataset de test dans le formulaire Streamlit.

    Cette fonction récupère une ligne spécifique du DataFrame de test et
    met à jour les valeurs du `session_state` afin de préremplir les sliders
    du formulaire avec les données correspondantes.

    Args:
        df (pd.DataFrame):
            Le DataFrame contenant le dataset de test.
        row_index (int):
            L'index de la ligne à charger dans le formulaire.

    Returns:
        None
    """
    row = df.iloc[row_index]

    st.session_state["sepal_length"] = float(row["sepal_length"])
    st.session_state["sepal_width"] = float(row["sepal_width"])
    st.session_state["petal_length"] = float(row["petal_length"])
    st.session_state["petal_width"] = float(row["petal_width"])


# -----------------------------
# MAIN
# -----------------------------
def main() -> None:
    """Lance l'application principale Streamlit.

    Cette fonction orchestre l'ensemble de l'interface utilisateur :
    - initialise l'état de session
    - charge le dataset de test
    - affiche les contrôles dans la sidebar
    - permet la saisie manuelle ou le chargement d'une ligne de test
    - envoie les données à l'API `/predict`
    - affiche la prédiction et la version active du modèle

    Returns:
        None
    """
    init_session_state()

    # Chargement dataset test
    test_df = load_test_dataset()

    # -----------------------------
    # HEADER
    # -----------------------------
    st.title("🌸 Iris Prediction App")
    st.caption(
        "Simulation d'une utilisation réelle avec MLflow Model Registry + alias Production"
    )

    # Bouton reload modèle
    col_reload1, col_reload2 = st.columns([1, 5])
    with col_reload1:
        if st.button("♻️ Recharger"):
            st.cache_resource.clear()
            st.rerun()

    st.divider()

    # -----------------------------
    # SIDEBAR
    # -----------------------------
    with st.sidebar:
        st.header("⚙️ Contrôle")

        st.markdown("### 📦 Modèle")

        if test_df is not None:
            st.markdown("### 🧪 Charger une ligne de test")

            selected_index = st.selectbox(
                "Choisir une ligne du dataset iris_test.csv",
                options=list(range(len(test_df))),
                format_func=lambda x: f"Ligne #{x}",
                key="selected_test_index",
            )

            if st.button("📥 Charger cette ligne"):
                load_test_row_into_form(test_df, selected_index)
                st.success(f"Ligne #{selected_index} chargée dans le formulaire")
                st.rerun()
        else:
            st.warning("⚠️ Fichier `data/iris_test.csv` introuvable")

    # -----------------------------
    # LAYOUT PRINCIPAL
    # -----------------------------
    left_col, right_col = st.columns([1.2, 1])

    # -----------------------------
    # FORMULAIRE
    # -----------------------------
    with left_col:
        st.subheader("📥 Saisie des caractéristiques")

        col1, col2 = st.columns(2)

        with col1:
            sepal_length = st.slider(
                "Sepal Length (cm)",
                min_value=4.0,
                max_value=8.0,
                step=0.1,
                key="sepal_length",
            )

            sepal_width = st.slider(
                "Sepal Width (cm)",
                min_value=2.0,
                max_value=4.5,
                step=0.1,
                key="sepal_width",
            )

        with col2:
            petal_length = st.slider(
                "Petal Length (cm)",
                min_value=1.0,
                max_value=7.0,
                step=0.1,
                key="petal_length",
            )

            petal_width = st.slider(
                "Petal Width (cm)",
                min_value=0.1,
                max_value=2.5,
                step=0.1,
                key="petal_width",
            )

        input_df = pd.DataFrame(
            [
                {
                    "sepal_length": sepal_length,
                    "sepal_width": sepal_width,
                    "petal_length": petal_length,
                    "petal_width": petal_width,
                }
            ]
        )

        st.markdown("### 📊 Données envoyées au modèle")
        st.dataframe(input_df, width="stretch")

        predict_clicked = st.button("🔮 Faire une prédiction", width="stretch")

    # -----------------------------
    # PREDICTION
    # -----------------------------
    with right_col:
        st.subheader("🤖 Résultat")

        if predict_clicked:
            try:
                payload = {
                    "sepal_length": sepal_length,
                    "sepal_width": sepal_width,
                    "petal_length": petal_length,
                    "petal_width": petal_width,
                }

                # URL de base de l'API (Docker ou exécution locale)
                api_base_url = os.getenv("STREAMLIT_API_URL", "http://api:9090")
                predict_url = f"{api_base_url}/predict"

                # Appel HTTP vers l'API FastAPI
                response = requests.post(predict_url, json=payload, timeout=10)

                if response.status_code == 200:
                    rep = response.json()
                    result = rep.get("prediction")
                    version = rep.get("version", "?")

                    # Bannière / badge version modèle
                    st.success(
                        f"🟢 Modèle en ligne : **iris_model@Production** | "
                        f"**Version active : v{version}**"
                    )

                    pred_label = CLASS_NAMES.get(result)
                    st.metric("Espèce prédite", pred_label)
                    st.success("✅ Prédiction effectuée")
                else:
                    st.error("L'API a renvoyé une erreur.")

            except Exception:
                st.error("Impossible de contacter l'api")
        else:
            st.info("Clique sur **Faire une prédiction** pour voir le résultat")

    st.divider()


if __name__ == "__main__":
    main()