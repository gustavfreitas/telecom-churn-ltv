"""Configurações centrais: caminhos, seed e listas de colunas."""

from pathlib import Path

# Raiz do projeto (config.py -> src -> raiz)
RAIZ = Path(__file__).resolve().parent.parent

DATA_PATH = RAIZ / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELOS_DIR = RAIZ / "models"
FIGURAS_DIR = RAIZ / "reports" / "figures"
METRICAS_PATH = RAIZ / "reports" / "metrics.json"

RANDOM_STATE = 42

SERVICOS_INTERNET = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
COLUNAS_BINARIAS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "MultipleLines",
    *SERVICOS_INTERNET,
]
COLS_CAT = ["InternetService", "Contract", "PaymentMethod"]
