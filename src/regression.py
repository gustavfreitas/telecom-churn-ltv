"""Regressão Linear: prever MonthlyCharges."""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    cross_val_predict,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from src.config import AUDITORIA_PATH, COLS_CAT, MODELOS_DIR, RANDOM_STATE
from src.evaluation import salvar_figura
from src.features import montar_preprocessador

# Fora: Churn (outro alvo), TotalCharges (≈ tenure × MonthlyCharges = vazamento)
# e n_servicos_internet (soma exata das 6 colunas de serviço = colinearidade).
COLS_EXCLUIDAS = ["MonthlyCharges", "TotalCharges", "Churn", "n_servicos_internet"]
COLS_NUM = ["tenure"]


def avaliar_regressao(y_true, y_pred) -> dict:
    """Calcula MAE, RMSE e R²."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def _pipeline_regressao() -> Pipeline:
    return Pipeline(
        [
            ("prep", montar_preprocessador(COLS_NUM, COLS_CAT, escalar=True)),
            ("modelo", LinearRegression()),
        ]
    )


def auditar_cobranca(
    df: pd.DataFrame, ids: pd.Series | None = None, n: int = 25
) -> tuple[pd.DataFrame, dict]:
    """Clientes cuja mensalidade mais destoa do que o modelo de preços prevê.

    Cada previsão vem de um modelo que NÃO viu o cliente (validação cruzada
    sobre a base inteira), então o resíduo é uma medida honesta de desvio. Um
    desvio grande é um candidato a revisão de faturamento ou de desconto, não
    uma prova de erro: o modelo só enxerga os atributos disponíveis.
    """
    y = df["MonthlyCharges"]
    X = df.drop(columns=COLS_EXCLUIDAS)
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    previsto = cross_val_predict(_pipeline_regressao(), X, y, cv=cv)

    tabela = pd.DataFrame(
        {
            "MonthlyCharges": y,
            "previsto": previsto,
            "residuo": y - previsto,
            "tenure": df["tenure"],
            "Contract": df["Contract"],
            "InternetService": df["InternetService"],
            "n_servicos_internet": df["n_servicos_internet"],
        }
    )
    if ids is not None:
        tabela.insert(0, "customerID", ids.loc[tabela.index].to_numpy())
    residuo_abs = tabela["residuo"].abs()
    resumo = {
        "clientes": int(len(tabela)),
        "residuo_abs_mediano": float(residuo_abs.median()),
        "residuo_abs_maximo": float(residuo_abs.max()),
        "clientes_com_desvio_acima_de_3": int((residuo_abs > 3).sum()),
        "clientes_com_desvio_acima_de_5": int((residuo_abs > 5).sum()),
    }
    ordem = residuo_abs.sort_values(ascending=False).index[:n]
    return tabela.loc[ordem].reset_index(drop=True), resumo


def salvar_auditoria(tabela: pd.DataFrame, caminho=AUDITORIA_PATH) -> None:
    """CSV no formato que o Excel brasileiro abre direto (; e vírgula decimal)."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    tabela.round(4).to_csv(
        caminho, sep=";", decimal=",", index=False, encoding="utf-8-sig"
    )


def executar_regressao(df: pd.DataFrame) -> dict:
    """Treina, avalia, salva gráficos e modelo. Retorna as métricas."""
    y = df["MonthlyCharges"]
    X = df.drop(columns=COLS_EXCLUIDAS)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipe = _pipeline_regressao()
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)

    metricas = {
        "linear_teste": avaliar_regressao(y_te, pred),
        "baseline_media": avaliar_regressao(y_te, np.full(len(y_te), y_tr.mean())),
    }
    cv_r2 = cross_val_score(pipe, X, y, cv=5, scoring="r2")
    metricas["r2_cv5"] = {"media": float(cv_r2.mean()), "desvio": float(cv_r2.std())}

    # Gráficos: coeficientes e real vs. previsto
    coefs = pd.Series(
        pipe["modelo"].coef_, index=pipe["prep"].get_feature_names_out()
    ).sort_values()
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    coefs.plot.barh(ax=ax[0], title="Coeficientes da Regressão Linear")
    ax[1].scatter(y_te, pred, alpha=0.3)
    ax[1].plot([y_te.min(), y_te.max()], [y_te.min(), y_te.max()], "r--")
    ax[1].set(xlabel="Real", ylabel="Previsto", title="Real vs. Previsto (teste)")
    fig.tight_layout()
    salvar_figura(fig, "regressao_coef_real_previsto.png")

    MODELOS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODELOS_DIR / "regressao_linear.joblib")
    return metricas
