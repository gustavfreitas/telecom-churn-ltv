"""Testes da carga e limpeza de dados."""

import pandas as pd

from src.config import COLUNAS_BINARIAS, DATA_PATH


def test_csv_existe():
    assert DATA_PATH.exists(), f"CSV não encontrado em {DATA_PATH}"


def test_numero_de_linhas(df):
    assert len(df) == 7043


def test_customer_id_removido(df):
    assert "customerID" not in df.columns


def test_total_charges_numerico_e_sem_nulos(df):
    assert pd.api.types.is_numeric_dtype(df["TotalCharges"])
    assert df["TotalCharges"].isna().sum() == 0


def test_total_charges_vazio_vira_zero_apenas_com_tenure_zero(df):
    # As 11 strings vazias do CSV pertencem a clientes com tenure = 0
    zerados = df[df["TotalCharges"] == 0]
    assert len(zerados) == 11
    assert (zerados["tenure"] == 0).all()


def test_sem_nulos_no_dataset(df):
    assert df.isna().sum().sum() == 0


def test_churn_binario_e_taxa_esperada(df):
    assert set(df["Churn"].unique()) == {0, 1}
    assert 0.25 < df["Churn"].mean() < 0.28  # ~26,5%


def test_colunas_binarias_sao_zero_ou_um(df):
    for col in COLUNAS_BINARIAS:
        assert set(df[col].unique()) <= {0, 1}, f"{col} não é binária"


def test_categorias_redundantes_removidas(df):
    textos = df.select_dtypes(exclude="number")
    for col in textos.columns:
        valores = set(df[col].unique())
        assert "No internet service" not in valores
        assert "No phone service" not in valores


def test_features_criadas(df):
    assert df["n_servicos_internet"].between(0, 6).all()
    assert set(df["pagamento_automatico"].unique()) <= {0, 1}
    assert set(df["tem_familia"].unique()) <= {0, 1}
