"""Configurações centrais: caminhos, seed e listas de colunas."""

from pathlib import Path

# Raiz do projeto (config.py -> src -> raiz)
RAIZ = Path(__file__).resolve().parent.parent

DATA_PATH = RAIZ / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODELOS_DIR = RAIZ / "models"
FIGURAS_DIR = RAIZ / "reports" / "figures"
METRICAS_PATH = RAIZ / "reports" / "metrics.json"
FILA_PATH = RAIZ / "reports" / "fila_retencao_teste.csv"
AUDITORIA_PATH = RAIZ / "reports" / "auditoria_cobranca.csv"
RESULTADOS_NEGOCIO_PATH = RAIZ / "reports" / "resultados_negocio.md"

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

# --- Premissas de negócio da retenção (ilustrativas: ajuste ao cenário real) ---
HORIZONTE_MESES = 12  # valor em risco de um cliente = mensalidade x horizonte
TAXA_SUCESSO_RETENCAO = 0.30  # fração dos que cancelariam que a oferta consegue reter
CUSTO_ACAO_RETENCAO = 20.0  # custo de contatar/ofertar a um cliente (US$)
GRADE_CUSTOS = (5, 20, 50, 100, 200)  # varredura de sensibilidade (US$ por ação)
GRADE_TAXAS = (0.10, 0.20, 0.30, 0.50)  # varredura de sensibilidade (taxa de sucesso)
