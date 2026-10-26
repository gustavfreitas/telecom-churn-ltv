"""Testes da camada de custo de negócio, da fila de retenção e da auditoria."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.negocio import (
    CalibradorPlatt,
    Premissas,
    beneficio_liquido,
    executar_analise_negocio,
    ganho_esperado,
    montar_fila_retencao,
    salvar_fila,
)
from src.regression import auditar_cobranca

P = Premissas(horizonte_meses=12, taxa_sucesso=0.3, custo_acao=20.0)


# ---------------------------------------------------------------------------
# Contas simples, conferíveis à mão
# ---------------------------------------------------------------------------


def test_ganho_esperado_conta_manual():
    # 0,5 x 0,3 x (100 x 12) - 20 = 160
    assert ganho_esperado([0.5], [100.0], P)[0] == pytest.approx(160.0)


def test_ganho_negativo_para_cliente_barato_e_improvavel():
    assert ganho_esperado([0.05], [20.0], P)[0] < 0


def test_beneficio_liquido_conta_manual():
    y = [1, 0, 1]
    contatar = [True, True, False]
    mensalidade = [100.0, 100.0, 100.0]
    # cliente 1: 0,3 x 1200 - 20 = 340; cliente 2: -20; cliente 3 não contatado
    assert beneficio_liquido(y, contatar, mensalidade, P) == pytest.approx(320.0)


def test_nao_contatar_ninguem_vale_zero():
    assert beneficio_liquido([1, 1], [False, False], [50.0, 80.0], P) == 0.0


# ---------------------------------------------------------------------------
# Calibração
# ---------------------------------------------------------------------------


def test_calibrador_reduz_brier_e_preserva_a_ordem():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.25, 4000)
    # score bom, mas inflado: probabilidades sistematicamente altas demais
    verdadeira = np.where(y == 1, 0.6, 0.2) + rng.normal(0, 0.05, 4000)
    inflada = np.clip(verdadeira + 0.25, 0.01, 0.99)

    cal = CalibradorPlatt().ajustar(inflada, y)
    ajustada = cal.transformar(inflada)

    assert brier_score_loss(y, ajustada) < brier_score_loss(y, inflada)
    assert abs(ajustada.mean() - y.mean()) < abs(inflada.mean() - y.mean())
    assert roc_auc_score(y, ajustada) == pytest.approx(roc_auc_score(y, inflada))


# ---------------------------------------------------------------------------
# Pipeline completo (usa os dados reais; módulo inteiro compartilha o resultado)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def analise(df, tmp_path_factory):
    from src.classification import executar_classificacao

    tmp = tmp_path_factory.mktemp("saidas")
    mp = pytest.MonkeyPatch()
    mp.setattr("src.evaluation.FIGURAS_DIR", tmp)
    mp.setattr("src.classification.MODELOS_DIR", tmp)
    try:
        res = executar_classificacao(df)
        neg = executar_analise_negocio(res)
    finally:
        mp.undo()
    return res, neg


def test_estrategias_respeitam_os_limites_teoricos(analise):
    _, neg = analise
    est = neg.estrategias.set_index("estrategia")["beneficio_total"]
    assert est["Não contatar ninguém"] == 0.0
    oraculo = est["Oráculo (só quem cancelaria; limite teórico)"]
    assert oraculo >= est.max() - 1e-6
    # a decisão por ganho esperado não deve perder para a regra antiga
    assert (
        est["XGBoost calibrado + ganho esperado > 0"]
        >= est["Regra antiga: probabilidade bruta do XGBoost ≥ 0,5"]
    )


def test_sensibilidade_custo_alto_contata_menos(analise):
    _, neg = analise
    contatados = neg.sens_custo["contatados_modelo"].to_numpy()
    assert np.all(np.diff(contatados) <= 0)


def test_fila_ordenada_e_coerente(analise, df):
    res, neg = analise
    ids = pd.Series([f"C{i}" for i in range(len(df))], index=df.index)
    fila = montar_fila_retencao(res, neg, ids)

    assert len(fila) == len(res.y_te)
    assert fila["ganho_esperado"].is_monotonic_decreasing
    assert (fila["contatar"] == (fila["ganho_esperado"] > 0)).all()
    assert fila["customerID"].is_unique


def test_salvar_fila_usa_formato_brasileiro(analise, df, tmp_path):
    res, neg = analise
    ids = pd.Series([f"C{i}" for i in range(len(df))], index=df.index)
    caminho = tmp_path / "fila.csv"
    salvar_fila(montar_fila_retencao(res, neg, ids), caminho)

    cabecalho, primeira = caminho.read_text(encoding="utf-8-sig").splitlines()[:2]
    assert ";" in cabecalho
    assert "," in primeira  # decimal com vírgula


def test_motivos_apontam_variaveis_reais(analise):
    pytest.importorskip("shap")
    from src.interpretability import motivos_por_cliente

    res, _ = analise
    motivos = motivos_por_cliente(res, n_motivos=3)
    assert len(motivos) == len(res.y_te)
    colunas = set(res.X_te.columns)
    amostra = motivos.iloc[0].split("; ")
    assert 1 <= len(amostra) <= 3
    assert all(parte.split("=")[0] in colunas for parte in amostra)


# ---------------------------------------------------------------------------
# Auditoria de cobrança
# ---------------------------------------------------------------------------


def test_auditoria_devolve_maiores_desvios_ordenados(df):
    ids = pd.Series([f"C{i}" for i in range(len(df))], index=df.index)
    tabela, resumo = auditar_cobranca(df, ids, n=10)

    assert len(tabela) == 10
    absolutos = tabela["residuo"].abs()
    assert absolutos.is_monotonic_decreasing
    assert resumo["residuo_abs_maximo"] == pytest.approx(absolutos.iloc[0])
    assert resumo["clientes"] == len(df)
