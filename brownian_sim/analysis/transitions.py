"""Matrice de tranziții între piese + timpi medii de rezidență."""
from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np
import pandas as pd


def load_transitions(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def transition_matrix(
    df: pd.DataFrame, piece_names: List[str]
) -> Tuple[np.ndarray, List[str]]:
    """Matricea MxM a numărului de tranziții piesă->piesă.

    Rânduri = from, coloane = to. Include "OUT" ca etichetă dacă apare.
    """
    labels = list(piece_names)
    if "OUT" in df["from"].values or "OUT" in df["to"].values:
        labels = labels + ["OUT"]
    idx = {name: k for k, name in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for _, row in df.iterrows():
        fr, to = row["from"], row["to"]
        if fr in idx and to in idx:
            M[idx[fr], idx[to]] += 1
    return M, labels


def residence_counts(
    piece_counts_csv: str,
    piece_names: List[str],
) -> pd.DataFrame:
    """Ocuparea medie per piesă pe parcursul rulării."""
    df = pd.read_csv(piece_counts_csv)
    return df[["step"] + piece_names].copy()
