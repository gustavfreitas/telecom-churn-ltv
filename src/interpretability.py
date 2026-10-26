"""Interpretabilidade: importância das variáveis e churn por segmento."""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

from src.classification import ResultadoClassificacao
from src.config import COLS_CAT, COLUNAS_BINARIAS, RANDOM_STATE
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


def _coluna_original(nome: str) -> str:
    """'Contract_One year' -> 'Contract' (agrupa as colunas do One-Hot)."""
    for coluna in COLS_CAT:
        if nome.startswith(f"{coluna}_"):
            return coluna
    return nome


def _formatar_valor(coluna: str, valor) -> str:
    if coluna == "gender":
        return "Male" if valor == 1 else "Female"
    if coluna in COLUNAS_BINARIAS:
        return "Sim" if valor == 1 else "Não"
    if isinstance(valor, str):
        return valor
    return str(int(valor)) if float(valor).is_integer() else f"{valor:.2f}"


def shap_por_coluna(res: ResultadoClassificacao) -> pd.DataFrame | None:
    """Contribuição SHAP de cada coluna original para o risco de cada cliente.

    Soma as colunas do One-Hot de volta na variável de origem. Devolve None se o
    pacote `shap` não estiver instalado (é uma dependência opcional).
    """
    try:
        import shap
    except ImportError:
        return None
    prep, modelo = res.pipeline["prep"], res.pipeline["modelo"]
    X_p = prep.transform(res.X_te)
    valores = shap.TreeExplainer(modelo).shap_values(X_p)
    tabela = pd.DataFrame(valores, index=res.X_te.index, columns=X_p.columns)
    return tabela.T.groupby(_coluna_original).sum().T


def motivos_por_cliente(
    res: ResultadoClassificacao, n_motivos: int = 3
) -> pd.Series | None:
    """Texto com as variáveis que mais empurram cada cliente para o cancelamento.

    Ex.: 'Contract=Month-to-month (+0,93); tenure=2 (+0,71)'. Os valores são
    contribuições SHAP (escala log-odds do modelo bruto): indicam a direção e o
    peso relativo, não uma probabilidade.
    """
    shap_col = shap_por_coluna(res)
    if shap_col is None:
        return None
    textos = []
    for idx, linha in shap_col.iterrows():
        partes = []
        for coluna, valor in linha.nlargest(n_motivos).items():
            if valor <= 0:
                continue
            valor_cliente = _formatar_valor(coluna, res.X_te.loc[idx, coluna])
            partes.append(f"{coluna}={valor_cliente} (+{valor:.2f})".replace(".", ","))
        textos.append("; ".join(partes))
    return pd.Series(textos, index=shap_col.index, name="motivos")
