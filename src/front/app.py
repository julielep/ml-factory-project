import sys
from pathlib import Path

import requests

# Ajoute la racine du projet au PYTHONPATH
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import streamlit as st

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(
    page_title="Iris Prediction App",
    page_icon="🌸",
    layout="wide"
)

# -----------------------------
# CONSTANTES
# -----------------------------
CLASS_NAMES = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

TEST_DATA_PATH = ROOT_DIR / "data" / "iris_test.csv"

# -----------------------------
# CHARGEMENT MODELE
# -----------------------------
# @st.cache_resource
# def get_model():
#     model, version = load_production_model()
#     return model, version


# -----------------------------
# CHARGEMENT DATASET DE TEST
# -----------------------------
@st.cache_data
def load_test_dataset():
    if TEST_DATA_PATH.exists():
        return pd.read_csv(TEST_DATA_PATH)
    return None


# -----------------------------
# INITIALISATION SESSION STATE
# -----------------------------
def init_session_state():
    defaults = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
        "selected_test_index": 0
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# -----------------------------
# CHARGER UNE LIGNE DE TEST
# -----------------------------
def load_test_row_into_form(df, row_index):
    row = df.iloc[row_index]

    st.session_state["sepal_length"] = float(row["sepal_length"])
    st.session_state["sepal_width"] = float(row["sepal_width"])
    st.session_state["petal_length"] = float(row["petal_length"])
    st.session_state["petal_width"] = float(row["petal_width"])


# -----------------------------
# MAIN
# -----------------------------
def main():
    init_session_state()

    # # Chargement modèle
    # try:
    #     model, version = get_model()
    # except Exception as e:
    #     st.error(f"❌ Erreur lors du chargement du modèle : {e}")
    #     st.stop()

    # Chargement dataset test
    test_df = load_test_dataset()

    # -----------------------------
    # HEADER
    # -----------------------------
    st.title("🌸 Iris Prediction App")
    st.caption("Simulation d'une utilisation réelle avec MLflow Model Registry + alias Production")


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
        # st.info(f"Alias : Production\n\nVersion : v{version}")

        if test_df is not None:
            st.markdown("### 🧪 Charger une ligne de test")

            selected_index = st.selectbox(
                "Choisir une ligne du dataset iris_test.csv",
                options=list(range(len(test_df))),
                format_func=lambda x: f"Ligne #{x}",
                key="selected_test_index"
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
                key="sepal_length"
            )

            sepal_width = st.slider(
                "Sepal Width (cm)",
                min_value=2.0,
                max_value=4.5,
                step=0.1,
                key="sepal_width"
            )

        with col2:
            petal_length = st.slider(
                "Petal Length (cm)",
                min_value=1.0,
                max_value=7.0,
                step=0.1,
                key="petal_length"
            )

            petal_width = st.slider(
                "Petal Width (cm)",
                min_value=0.1,
                max_value=2.5,
                step=0.1,
                key="petal_width"
            )

        input_df = pd.DataFrame([{
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width
        }])

        st.markdown("### 📊 Données envoyées au modèle")
        st.dataframe(input_df, use_container_width=True)

        predict_clicked = st.button("🔮 Faire une prédiction", use_container_width=True)

    # -----------------------------
    # PREDICTION + PROBAS
    # -----------------------------
    with right_col:
        st.subheader("🤖 Résultat")

        if predict_clicked:
            try:
                features = [[sepal_length, sepal_width, petal_length, petal_width]]
                # Prédiction
                payload = {
                    "sepal_length": sepal_length,
                    "sepal_width": sepal_width,
                    "petal_length": petal_length,
                    "petal_width": petal_width
                }
                # Appeler l'API (utilise le nom du service Docker ou localhost:9090)
                # Note: depuis Windows, utilise localhost:9090. Depuis Docker, utilise api:9090.
                response = requests.post("http://api:9090/predict", json=payload)
                
                if response.status_code == 200:
                    rep = response.json()
                    result = rep.get('prediction') 
                    version = rep.get('version', '?') 
                    # Bannière / badge version modèle
                    st.success(f"🟢 Modèle en ligne : **iris_model@Production** | **Version active : v{version}**")
                    pred_label = CLASS_NAMES.get(result)
                    st.metric("Espèce prédite", pred_label)
                else:
                    st.error("L'API a renvoyé une erreur.")

                st.success("✅ Prédiction effectuée")
    

            except Exception:
                st.error("Impossible de contacter l'api")
        else:
            st.info("Clique sur **Faire une prédiction** pour voir le résultat")

    st.divider()

    # # -----------------------------
    # # APERÇU DATASET TEST
    # # -----------------------------
    # st.subheader("🧪 Aperçu du dataset de test")

    # if test_df is not None:
    #     st.dataframe(test_df.head(10), use_container_width=True)
    # else:
    #     st.warning("Le fichier `data/iris_test.csv` n'a pas été trouvé.")

    # # -----------------------------
    # # FOOTER TECHNIQUE
    # # -----------------------------
    # with st.expander("ℹ️ Informations techniques"):
    #     st.write("**Nom du modèle :** iris_model")
    #     st.write("**Alias utilisé :** Production")
    #     st.write(f"**Version active :** v{version}")
    #     st.write("**Source du modèle :** MLflow Model Registry + MinIO")
    #     st.write("**Dataset de test :** data/iris_test.csv")


if __name__ == "__main__":
    main()