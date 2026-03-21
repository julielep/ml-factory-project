"""Tests unitaires du module src.front.app."""

from unittest.mock import patch

import pandas as pd
import streamlit as st

from src.front.app import load_test_dataset


@patch("src.front.app.TEST_DATA_PATH")
@patch("src.front.app.pd.read_csv")
def test_load_test_dataset_file_exists(mock_read_csv, mock_test_data_path):
    """Vérifie que le dataset est chargé si le fichier existe."""
    mock_test_data_path.exists.return_value = True
    fake_df = pd.DataFrame({"sepal_length": [5.1]})
    mock_read_csv.return_value = fake_df

    result = load_test_dataset()

    assert result is fake_df
    mock_read_csv.assert_called_once()


@patch("src.front.app.TEST_DATA_PATH")
def test_load_test_dataset_file_missing(mock_test_data_path):
    """Vérifie que None est retourné si le fichier n'existe pas."""
    st.cache_data.clear()

    mock_test_data_path.exists.return_value = False

    result = load_test_dataset()

    assert result is None
