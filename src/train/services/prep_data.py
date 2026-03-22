"""src/train/services/prep_data.py"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def prepare_data():
    """Charge le dataset Iris et le découpe en ensembles d'entraînement et de test.

    Cette fonction récupère le dataset Iris depuis scikit-learn, extrait
    les variables explicatives (`X`) et la variable cible (`y`), puis
    applique un découpage train/test avec une proportion de 80/20.

    Returns:
        tuple: Un tuple contenant :
            - X_train : Features d'entraînement
            - X_test : Features de test
            - y_train : Labels d'entraînement
            - y_test : Labels de test
    """
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test
