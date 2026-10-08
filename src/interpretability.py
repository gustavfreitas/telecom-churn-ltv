"""Interpretabilidade: importância das variáveis e churn por segmento."""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from src.classification import ResultadoClassificacao
from src.config import RANDOM_STATE
from src.evaluation import salvar_figura


def gerar_importancias(res: ResultadoClassificacao, top: int = 15) -> pd.DataFrame:
    """Importância por Gain (nativa) e por Permutation (queda de AUC no teste)."""
    prep, modelo = res.pipeline["prep"], res.pipeline["modelo"]
    X_te_p = prep.transform(res.X_te)
    nomes = prep.get_feature_names_out()

    gain = (
        pd.Series(modelo.get_booster().get_score(importance_type="gain"))
        .reindex(nomes)
        .fillna(0)
    )
    perm = permutation_importance(
        modelo,
        X_te_p,
        res.y_te,
        scoring="roc_auc",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    tabela = pd.DataFrame(
        {"gain": gain, "permutation_auc": pd.Series(perm.importances_mean, index=nomes)}
    )

    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
    tabela["gain"].sort_values().tail(top).plot.barh(
        ax=ax[0], title=f"Top {top} - Importância (Gain)"
    )
    tabela["permutation_auc"].sort_values().tail(top).plot.barh(
        ax=ax[1], title=f"Top {top} - Permutation (queda de AUC)"
    )
    fig.tight_layout()
    salvar_figura(fig, "importancia_variaveis.png")
    return tabela.sort_values("permutation_auc", ascending=False)


def churn_por_segmento(df: pd.DataFrame, colunas: list[str]) -> dict:
    """Taxa real de churn por categoria de cada coluna informada."""
    return {
        c: df.groupby(c)["Churn"].mean().sort_values(ascending=False).round(3)
        for c in colunas
    }


def top_risco(res: ResultadoClassificacao, n: int = 10) -> pd.DataFrame:
    """Clientes do teste com maior probabilidade de churn (lista de retenção)."""
    risco = res.X_te.assign(prob_churn=res.proba, churn_real=res.y_te.values)
    cols = ["tenure", "Contract", "MonthlyCharges", "prob_churn", "churn_real"]
    return risco.sort_values("prob_churn", ascending=False).head(n)[cols]
