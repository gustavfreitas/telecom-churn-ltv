"""Regressão Linear: prever MonthlyCharges."""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.config import COLS_CAT, MODELOS_DIR, RANDOM_STATE
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


def executar_regressao(df: pd.DataFrame) -> dict:
    """Treina, avalia, salva gráficos e modelo. Retorna as métricas."""
    y = df["MonthlyCharges"]
    X = df.drop(columns=COLS_EXCLUIDAS)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    pipe = Pipeline(
        [
            ("prep", montar_preprocessador(COLS_NUM, COLS_CAT, escalar=True)),
            ("modelo", LinearRegression()),
        ]
    )
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)

    metricas = {
        "linear_teste": avaliar_regressao(y_te, pred),
        "baseline_media": avaliar_regressao(
            y_te, np.full(len(y_te), y_tr.mean())
        ),
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
