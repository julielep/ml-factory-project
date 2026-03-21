"""Tests unitaires du module src.train.train."""

from src.train.train import prepare_data


def test_prepare_data_returns_four_objects():
    """Vérifie que prepare_data retourne bien 4 objets."""
    result = prepare_data()

    assert isinstance(result, tuple)
    assert len(result) == 4


def test_prepare_data_split_sizes():
    """Vérifie que le split train/test respecte bien 80/20 sur Iris."""
    X_train, X_test, y_train, y_test = prepare_data()

    # Dataset Iris = 150 lignes
    assert len(X_train) == 120
    assert len(X_test) == 30
    assert len(y_train) == 120
    assert len(y_test) == 30


def test_prepare_data_feature_count():
    """Vérifie que les données contiennent bien 4 features."""
    X_train, X_test, _, _ = prepare_data()

    assert X_train.shape[1] == 4
    assert X_test.shape[1] == 4
