"""Unit tests for src/data_loader.py in Phase 1."""

import pandas as pd
import pytest
from src.data_loader import standardize_dataframe, REQUIRED_COLUMNS, FTR_MAPPING


def test_standardize_dataframe():
    sample_raw = pd.DataFrame({
        "Date": ["12/09/2020", "13/09/2020", "14/09/2020"],
        "HomeTeam": ["Fulham", "West Brom", "Brighton"],
        "AwayTeam": ["Arsenal", "Leicester", "Chelsea"],
        "FTHG": [0, 0, 1],
        "FTAG": [3, 3, 3],
        "FTR": ["A", "A", "A"],
        "HS": [5, 7, 13],
        "AS": [13, 15, 10],
        "HST": [2, 1, 3],
        "AST": [6, 7, 5],
        "Referee": ["C Kavanagh", "A Taylor", "C Pawson"],  # Extra column to ignore
    })

    cleaned = standardize_dataframe(sample_raw)

    # Verify columns exactly match required schema
    assert list(cleaned.columns) == REQUIRED_COLUMNS

    # Verify FTR mapping (A -> 0)
    assert (cleaned["FTR"] == 0).all()

    # Verify Date parsing
    assert pd.api.types.is_datetime64_any_dtype(cleaned["Date"])

    # Verify data types
    assert pd.api.types.is_integer_dtype(cleaned["FTR"])
    assert pd.api.types.is_integer_dtype(cleaned["FTHG"])
    assert pd.api.types.is_integer_dtype(cleaned["HST"])


def test_standardize_target_mapping():
    sample_raw = pd.DataFrame({
        "Date": ["01/01/2021", "02/01/2021", "03/01/2021"],
        "HomeTeam": ["Arsenal", "Chelsea", "Liverpool"],
        "AwayTeam": ["Chelsea", "Arsenal", "Man United"],
        "FTHG": [2, 1, 0],
        "FTAG": [1, 1, 1],
        "FTR": ["H", "D", "A"],
        "HS": [10, 12, 8],
        "AS": [8, 9, 14],
        "HST": [5, 4, 2],
        "AST": [3, 4, 6],
    })

    cleaned = standardize_dataframe(sample_raw)
    assert list(cleaned["FTR"]) == [2, 1, 0]


def test_standardize_drops_invalid_rows():
    sample_raw = pd.DataFrame({
        "Date": ["01/01/2021", None, "03/01/2021"],
        "HomeTeam": ["Arsenal", "Chelsea", "Liverpool"],
        "AwayTeam": ["Chelsea", "Arsenal", None],
        "FTHG": [2, None, 0],
        "FTAG": [1, 1, 1],
        "FTR": ["H", "D", "A"],
        "HS": [10, 12, 8],
        "AS": [8, 9, 14],
        "HST": [5, 4, 2],
        "AST": [3, 4, 6],
    })

    cleaned = standardize_dataframe(sample_raw)
    # Rows with null Date or null AwayTeam must be dropped
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["HomeTeam"] == "Arsenal"
