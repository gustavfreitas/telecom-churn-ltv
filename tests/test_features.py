"""Testes do pré-processamento e da prevenção de vazamento (data leakage)."""

import numpy as np
import pandas as pd
import pytest

from src.classification import COLS_NUM as NUM_CLF
from src.config import COLS_CAT
from src.features import montar_preprocessador
from src.regression import COLS_EXCLUIDAS
from src.regression import COLS_NUM as NUM_REG


def test_regressao_exclui_colunas_com_vazamento(df):
    X = df.drop(columns=COLS_EXCLUIDAS)
    for col in ["MonthlyCharges", "TotalCharges", "Churn", "n_servicos_internet"]:
        assert col not in X.columns


def test_preprocessador_devolve_dataframe_sem_nulos(df):
    X = df.drop(columns=COLS_EXCLUIDAS)
    saida = montar_preprocessador(NUM_REG, COLS_CAT, escalar=True).fit_transform(X)
    assert isinstance(saida, pd.DataFrame)
    assert saida.isna().sum().sum() == 0
    assert len(saida) == len(X)


def test_one_hot_nao_deixa_colunas_texto(df):
    X = df.drop(columns=["Churn"])
    saida = montar_preprocessador(NUM_CLF, COLS_CAT, escalar=False).fit_transform(X)
    assert saida.select_dtypes(exclude="number").empty


def test_escala_padroniza_tenure(df):
    X = df.drop(columns=COLS_EXCLUIDAS)
    saida = montar_preprocessador(NUM_REG, COLS_CAT, escalar=True).fit_transform(X)
    assert abs(saida["tenure"].mean()) < 1e-6
    assert abs(saida["tenure"].std(ddof=0) - 1) < 1e-6


def test_sem_escala_preserva_valores_originais(df):
    X = df.drop(columns=["Churn"])
    saida = montar_preprocessador(NUM_CLF, COLS_CAT, escalar=False).fit_transform(X)
    assert np.array_equal(saida["tenure"].to_numpy(), X["tenure"].to_numpy())


def test_categoria_desconhecida_nao_quebra(df):
    # handle_unknown="ignore": categoria nova em produção vira linha de zeros
    X = df.drop(columns=COLS_EXCLUIDAS)
    prep = montar_preprocessador(NUM_REG, COLS_CAT, escalar=True).fit(X)
    novo = X.head(3).copy()
    novo["Contract"] = "Contrato inexistente"
    with pytest.warns(UserWarning, match="unknown categories"):
        saida = prep.transform(novo)
    assert saida.isna().sum().sum() == 0
