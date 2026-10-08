# XGBoost Classifier: prever Churn (com baseline de Regressão Logística).

from dataclasses import dataclass

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import COLS_CAT, MODELOS_DIR, RANDOM_STATE
from src.evaluation import salvar_figura
from src.features import montar_preprocessador

COLS_NUM = ["tenure", "MonthlyCharges", "TotalCharges", "n_servicos_internet"]
LIMIAR = 0.5  # limiar de decisão; ajustável conforme o custo de negócio


@dataclass
class ResultadoClassificacao:
    """Artefatos necessários para a etapa de interpretabilidade."""

    metricas: dict
    pipeline: Pipeline
    X_te: pd.DataFrame
    y_te: pd.Series
    proba: pd.Series


def executar_classificacao(df: pd.DataFrame) -> ResultadoClassificacao:
    """Treina XGBoost e baseline, avalia, salva gráficos e modelo."""
    y = df["Churn"]
    X = df.drop(columns=["Churn"])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Desbalanceamento (~27% churn): peso da classe positiva = neg / pos
    spw = float((y_tr == 0).sum() / (y_tr == 1).sum())

    # Pipeline completo: o pré-processador é ajustado dentro de cada fold
    pipe_xgb = Pipeline(
        [
            ("prep", montar_preprocessador(COLS_NUM, COLS_CAT, escalar=False)),
            (
                "modelo",
                XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=3,  # árvores rasas: dataset pequeno
                    subsample=0.8,
                    colsample_bytree=0.8,
                    min_child_weight=3,
                    scale_pos_weight=spw,
                    eval_metric="auc",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    # Baseline: Regressão Logística (precisa de escala)
    pipe_log = Pipeline(
        [
            ("prep", montar_preprocessador(COLS_NUM, COLS_CAT, escalar=True)),
            (
                "modelo",
                LogisticRegression(
                    max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
                ),
            ),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    metricas = {"scale_pos_weight": spw, "limiar": LIMIAR}
    for nome, pipe in [("xgboost", pipe_xgb), ("regressao_logistica", pipe_log)]:
        cv_auc = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc")
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        pred = (proba >= LIMIAR).astype(int)
        relatorio = classification_report(
            y_te, pred, target_names=["Ficou", "Cancelou"], output_dict=True
        )
        metricas[nome] = {
            "auc_cv5_media": float(cv_auc.mean()),
            "auc_cv5_desvio": float(cv_auc.std()),
            "auc_teste": float(roc_auc_score(y_te, proba)),
            "churn_precision": relatorio["Cancelou"]["precision"],
            "churn_recall": relatorio["Cancelou"]["recall"],
            "churn_f1": relatorio["Cancelou"]["f1-score"],
        }
        if nome == "xgboost":
            print(classification_report(y_te, pred, target_names=["Ficou", "Cancelou"]))
            proba_xgb, pred_xgb = proba, pred

    # Gráficos do XGBoost: matriz de confusão e ROC
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_te, pred_xgb, ax=ax[0], colorbar=False)
    RocCurveDisplay.from_predictions(y_te, proba_xgb, ax=ax[1], name="XGBoost")
    RocCurveDisplay.from_estimator(
        pipe_log, X_te, y_te, ax=ax[1], name="Regressão Logística"
    )
    ax[1].plot([0, 1], [0, 1], "k--")
    fig.tight_layout()
    salvar_figura(fig, "classificacao_confusao_roc.png")

    MODELOS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe_xgb, MODELOS_DIR / "xgboost_churn.joblib")
    return ResultadoClassificacao(metricas, pipe_xgb, X_te, y_te, proba_xgb)
