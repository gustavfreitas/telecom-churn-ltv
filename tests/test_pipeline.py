#Teste de fumaça: o pipeline roda de ponta a ponta numa amostra pequena.

import pytest

from src.classification import executar_classificacao
from src.interpretability import (
    churn_por_segmento,
    gerar_importancias,
    top_risco,
)
from src.regression import executar_regressao


@pytest.fixture()
def saidas_temporarias(tmp_path, monkeypatch):
    #Redireciona figuras e modelos para a pasta temporária do teste.
    monkeypatch.setattr("src.evaluation.FIGURAS_DIR", tmp_path / "figuras")
    monkeypatch.setattr("src.regression.MODELOS_DIR", tmp_path / "modelos")
    monkeypatch.setattr("src.classification.MODELOS_DIR", tmp_path / "modelos")
    return tmp_path


def test_regressao_roda_e_tem_bom_ajuste(df, saidas_temporarias):
    metricas = executar_regressao(df.sample(1500, random_state=0))
    assert metricas["linear_teste"]["R2"] > 0.99
    assert metricas["linear_teste"]["MAE"] < metricas["baseline_media"]["MAE"]
    assert (saidas_temporarias / "modelos" / "regressao_linear.joblib").exists()


def test_classificacao_e_interpretabilidade(df, saidas_temporarias):
    res = executar_classificacao(df.sample(1500, random_state=0))
    assert res.metricas["xgboost"]["auc_teste"] > 0.7
    assert 0 <= res.metricas["xgboost"]["churn_recall"] <= 1

    tabela = gerar_importancias(res, top=5)
    assert {"gain", "permutation_auc"} <= set(tabela.columns)

    riscos = top_risco(res, n=5)
    assert len(riscos) == 5
    assert riscos["prob_churn"].is_monotonic_decreasing

    seg = churn_por_segmento(df, ["Contract"])
    assert seg["Contract"].idxmax() == "Month-to-month"
