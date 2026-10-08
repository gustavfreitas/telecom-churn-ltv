"""Carga e limpeza do dataset Telco Customer Churn."""

from pathlib import Path

import pandas as pd

from src.config import COLUNAS_BINARIAS, DATA_PATH, SERVICOS_INTERNET


def carregar_e_limpar(caminho: Path = DATA_PATH) -> pd.DataFrame:
    """Lê o CSV, corrige tipos, trata nulos e cria novas features."""
    df = pd.read_csv(caminho)

    # TotalCharges vem como texto; 11 linhas são strings vazias (clientes com
    # tenure = 0, ainda sem fatura). Converter para número e preencher com 0.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # Categorias redundantes com InternetService / PhoneService -> "No"
    df = df.replace({"No internet service": "No", "No phone service": "No"})

    # Variáveis binárias (Yes/No, Male/Female) -> 0/1
    for col in COLUNAS_BINARIAS:
        df[col] = df[col].map({"Yes": 1, "No": 0, "Male": 1, "Female": 0})

    # --- Feature Engineering ---
    df["n_servicos_internet"] = df[SERVICOS_INTERNET].sum(axis=1)
    df["pagamento_automatico"] = (
        df["PaymentMethod"].str.contains("automatic").astype(int)
    )
    df["tem_familia"] = ((df["Partner"] == 1) | (df["Dependents"] == 1)).astype(int)

    # Alvo da classificação
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df.drop(columns=["customerID"])
