"""Custo de negócio da retenção: calibração, ganho esperado e estratégias.

Ideia central: contatar um cliente só vale a pena se

    prob_churn x taxa_de_sucesso x valor_em_risco > custo_da_ação

Isso exige probabilidades CALIBRADAS (o XGBoost com `scale_pos_weight` as infla).
Por isso o calibrador é ajustado só com previsões out-of-fold do TREINO; o teste
é usado apenas para medir. As premissas (custo, taxa de sucesso, horizonte) são
ilustrativas e ficam em `config.py`.
"""

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.classification import ResultadoClassificacao
from src.config import (
    CUSTO_ACAO_RETENCAO,
    FILA_PATH,
    GRADE_CUSTOS,
    GRADE_TAXAS,
    HORIZONTE_MESES,
    RANDOM_STATE,
    TAXA_SUCESSO_RETENCAO,
)
from src.evaluation import salvar_figura


@dataclass(frozen=True)
class Premissas:
    """Premissas econômicas da campanha de retenção."""

    horizonte_meses: int = HORIZONTE_MESES
    taxa_sucesso: float = TAXA_SUCESSO_RETENCAO
    custo_acao: float = CUSTO_ACAO_RETENCAO


class CalibradorPlatt:
    """Calibra probabilidades com uma regressão logística sobre o logit.

    É monotônico: muda a escala das probabilidades, não a ordem dos clientes
    (portanto não altera a AUC).
    """

    def __init__(self) -> None:
        self._modelo = LogisticRegression(C=1e6, max_iter=1000)

    @staticmethod
    def _logit(proba) -> np.ndarray:
        p = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def ajustar(self, proba, y) -> "CalibradorPlatt":
        self._modelo.fit(self._logit(proba), np.asarray(y))
        return self

    def transformar(self, proba) -> np.ndarray:
        return self._modelo.predict_proba(self._logit(proba))[:, 1]


def proba_out_of_fold(pipeline, X, y, dobras: int = 5) -> np.ndarray:
    """Probabilidade de cada linha prevista por um modelo que não a viu no treino."""
    cv = StratifiedKFold(n_splits=dobras, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]


# ---------------------------------------------------------------------------
# Valor e ganho
# ---------------------------------------------------------------------------


def valor_em_risco(mensalidade, premissas: Premissas) -> np.ndarray:
    """Receita em risco se o cliente cancelar: mensalidade x horizonte."""
    return np.asarray(mensalidade, dtype=float) * premissas.horizonte_meses


def ganho_esperado(proba, mensalidade, premissas: Premissas) -> np.ndarray:
    """Ganho líquido esperado de contatar cada cliente (pode ser negativo)."""
    receita_salva = (
        np.asarray(proba, dtype=float)
        * premissas.taxa_sucesso
        * valor_em_risco(mensalidade, premissas)
    )
    return receita_salva - premissas.custo_acao


def beneficio_liquido(y, contatar, mensalidade, premissas: Premissas) -> float:
    """Benefício total de uma estratégia, dado quem cancelou de fato.

    Quem cancelaria e é contatado gera `taxa_sucesso x valor_em_risco`; todo
    cliente contatado custa `custo_acao`.
    """
    y = np.asarray(y)
    contatar = np.asarray(contatar, dtype=bool)
    por_cliente = (
        y * premissas.taxa_sucesso * valor_em_risco(mensalidade, premissas)
        - premissas.custo_acao
    )
    return float(por_cliente[contatar].sum())


def comparar_estrategias(
    y, mensalidade, proba_bruta_xgb, proba_cal_xgb, proba_cal_log, premissas
) -> pd.DataFrame:
    """Compara quem contatar sob diferentes regras (no conjunto de teste)."""
    y = np.asarray(y)
    n = len(y)
    regras = {
        "Não contatar ninguém": np.zeros(n, dtype=bool),
        "Contatar todos": np.ones(n, dtype=bool),
        "Regra antiga: probabilidade bruta do XGBoost ≥ 0,5": (
            np.asarray(proba_bruta_xgb) >= 0.5
        ),
        "XGBoost calibrado + ganho esperado > 0": (
            ganho_esperado(proba_cal_xgb, mensalidade, premissas) > 0
        ),
        "Regressão Logística calibrada + ganho esperado > 0": (
            ganho_esperado(proba_cal_log, mensalidade, premissas) > 0
        ),
        "Oráculo (só quem cancelaria; limite teórico)": y == 1,
    }
    linhas = []
    for nome, mascara in regras.items():
        total = beneficio_liquido(y, mascara, mensalidade, premissas)
        linhas.append(
            {
                "estrategia": nome,
                "contatados": int(mascara.sum()),
                "cancelamentos_alcancados": int((mascara & (y == 1)).sum()),
                "beneficio_total": total,
                "beneficio_por_1000": total / n * 1000,
            }
        )
    return pd.DataFrame(linhas)


def _linha_sensibilidade(y, mensalidade, proba_bruta, proba_cal, premissas) -> dict:
    n = len(y)
    contatar_modelo = ganho_esperado(proba_cal, mensalidade, premissas) > 0
    por_1000 = 1000 / n
    return {
        "contatados_modelo": int(contatar_modelo.sum()),
        "modelo": beneficio_liquido(y, contatar_modelo, mensalidade, premissas)
        * por_1000,
        "regra_antiga": beneficio_liquido(
            y, np.asarray(proba_bruta) >= 0.5, mensalidade, premissas
        )
        * por_1000,
        "todos": beneficio_liquido(y, np.ones(n, dtype=bool), mensalidade, premissas)
        * por_1000,
    }


def sensibilidade(
    y, mensalidade, proba_bruta, proba_cal, base: Premissas, custos, taxas
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Benefício por 1.000 clientes variando o custo da ação e a taxa de sucesso."""
    y = np.asarray(y)
    por_custo = pd.DataFrame(
        [
            {
                "custo_acao": c,
                **_linha_sensibilidade(
                    y,
                    mensalidade,
                    proba_bruta,
                    proba_cal,
                    replace(base, custo_acao=c),
                ),
            }
            for c in custos
        ]
    )
    por_taxa = pd.DataFrame(
        [
            {
                "taxa_sucesso": t,
                **_linha_sensibilidade(
                    y,
                    mensalidade,
                    proba_bruta,
                    proba_cal,
                    replace(base, taxa_sucesso=t),
                ),
            }
            for t in taxas
        ]
    )
    return por_custo, por_taxa


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------


@dataclass
class ResultadoNegocio:
    premissas: Premissas
    calibracao: dict
    estrategias: pd.DataFrame
    sens_custo: pd.DataFrame
    sens_taxa: pd.DataFrame
    proba_cal: np.ndarray  # XGBoost calibrado, no teste


def _grafico_calibracao(y, bruta, calibrada) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for nome, proba in [("XGBoost bruto", bruta), ("XGBoost calibrado", calibrada)]:
        real, prevista = calibration_curve(y, proba, n_bins=8, strategy="quantile")
        ax.plot(prevista, real, marker="o", label=nome)
    ax.plot([0, 1], [0, 1], "k--", label="Calibração perfeita")
    ax.set(
        xlabel="Probabilidade prevista",
        ylabel="Fração real de cancelamentos",
        title="Calibração das probabilidades (teste)",
    )
    ax.legend()
    fig.tight_layout()
    salvar_figura(fig, "calibracao.png")


def _grafico_sensibilidade(sens_custo, sens_taxa) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for eixo, tabela, coluna, rotulo in [
        (ax[0], sens_custo, "custo_acao", "Custo de cada ação de retenção (US$, log)"),
        (ax[1], sens_taxa, "taxa_sucesso", "Taxa de sucesso da retenção"),
    ]:
        eixo.plot(
            tabela[coluna], tabela["modelo"], marker="o", label="Modelo calibrado"
        )
        eixo.plot(
            tabela[coluna],
            tabela["regra_antiga"],
            marker="s",
            label="Regra antiga (≥ 0,5)",
        )
        eixo.plot(tabela[coluna], tabela["todos"], marker="^", label="Contatar todos")
        eixo.axhline(0, color="gray", linestyle="--", label="Não contatar ninguém")
        eixo.set(xlabel=rotulo, ylabel="Benefício líquido por 1.000 clientes (US$)")
        eixo.legend()
    ax[0].set_xscale("log")
    fig.suptitle("Sensibilidade às premissas de negócio (teste)")
    fig.tight_layout()
    salvar_figura(fig, "sensibilidade_retencao.png")


def executar_analise_negocio(
    res: ResultadoClassificacao, premissas: Premissas | None = None
) -> ResultadoNegocio:
    """Calibra as probabilidades (só com o treino) e mede o benefício no teste."""
    premissas = premissas or Premissas()
    mensalidade = res.X_te["MonthlyCharges"].to_numpy()
    y_te = np.asarray(res.y_te)

    cal_xgb = CalibradorPlatt().ajustar(
        proba_out_of_fold(res.pipeline, res.X_tr, res.y_tr), res.y_tr
    )
    cal_log = CalibradorPlatt().ajustar(
        proba_out_of_fold(res.pipeline_log, res.X_tr, res.y_tr), res.y_tr
    )
    proba_cal_xgb = cal_xgb.transformar(res.proba)
    proba_cal_log = cal_log.transformar(res.proba_log)

    calibracao = {
        "churn_real_teste": float(y_te.mean()),
        "prob_media_xgb_bruta": float(np.mean(res.proba)),
        "prob_media_xgb_calibrada": float(proba_cal_xgb.mean()),
        "brier_xgb_bruto": float(brier_score_loss(y_te, res.proba)),
        "brier_xgb_calibrado": float(brier_score_loss(y_te, proba_cal_xgb)),
        "brier_log_bruta": float(brier_score_loss(y_te, res.proba_log)),
        "brier_log_calibrada": float(brier_score_loss(y_te, proba_cal_log)),
    }
    estrategias = comparar_estrategias(
        y_te, mensalidade, res.proba, proba_cal_xgb, proba_cal_log, premissas
    )
    sens_custo, sens_taxa = sensibilidade(
        y_te,
        mensalidade,
        res.proba,
        proba_cal_xgb,
        premissas,
        GRADE_CUSTOS,
        GRADE_TAXAS,
    )
    _grafico_calibracao(y_te, res.proba, proba_cal_xgb)
    _grafico_sensibilidade(sens_custo, sens_taxa)
    return ResultadoNegocio(
        premissas, calibracao, estrategias, sens_custo, sens_taxa, proba_cal_xgb
    )


def metricas_negocio(neg: ResultadoNegocio) -> dict:
    """Resumo serializável (vai para reports/metrics.json)."""
    return {
        "premissas": asdict(neg.premissas),
        "calibracao": neg.calibracao,
        "estrategias": neg.estrategias.to_dict("records"),
        "sensibilidade_custo": neg.sens_custo.to_dict("records"),
        "sensibilidade_taxa": neg.sens_taxa.to_dict("records"),
    }


# ---------------------------------------------------------------------------
# Fila de retenção
# ---------------------------------------------------------------------------


def montar_fila_retencao(
    res: ResultadoClassificacao,
    neg: ResultadoNegocio,
    ids: pd.Series,
    motivos: pd.Series | None = None,
) -> pd.DataFrame:
    """Clientes do teste ordenados por ganho esperado de contatá-los."""
    X = res.X_te
    mensalidade = X["MonthlyCharges"].to_numpy()
    fila = pd.DataFrame(
        {
            "customerID": ids.loc[X.index].to_numpy(),
            "prob_churn": neg.proba_cal,
            "valor_em_risco": valor_em_risco(mensalidade, neg.premissas),
            "ganho_esperado": ganho_esperado(neg.proba_cal, mensalidade, neg.premissas),
            "tenure": X["tenure"].to_numpy(),
            "Contract": X["Contract"].to_numpy(),
            "InternetService": X["InternetService"].to_numpy(),
            "MonthlyCharges": mensalidade,
        },
        index=X.index,
    )
    fila["contatar"] = fila["ganho_esperado"] > 0
    if motivos is not None:
        fila["motivos"] = motivos
    # Só existe porque é conjunto de teste (serve para validar a fila).
    fila["churn_real"] = np.asarray(res.y_te)
    return fila.sort_values("ganho_esperado", ascending=False).reset_index(drop=True)


def salvar_fila(fila: pd.DataFrame, caminho: Path = FILA_PATH) -> None:
    """CSV no formato que o Excel brasileiro abre direto (; e vírgula decimal)."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    fila.round(4).to_csv(
        caminho, sep=";", decimal=",", index=False, encoding="utf-8-sig"
    )


# ---------------------------------------------------------------------------
# Relatório em Markdown
# ---------------------------------------------------------------------------


def _dec(valor: float, casas: int = 3) -> str:
    return f"{valor:.{casas}f}".replace(".", ",")


def _int(valor: float) -> str:
    return f"{valor:,.0f}".replace(",", ".")


def _usd(valor: float) -> str:
    sinal = "-" if valor < 0 else ""
    return f"{sinal}US$ {_int(abs(valor))}"


def renderizar_markdown(neg: ResultadoNegocio) -> str:
    """Tabelas prontas para o README (todos os números vêm do pipeline)."""
    p, c = neg.premissas, neg.calibracao
    linhas = [
        "# Resultados de negócio (gerado por `python main.py`)",
        "",
        "Protocolo: o calibrador é ajustado só com previsões out-of-fold do "
        "**treino**; o teste é usado uma única vez, para medir. "
        "As premissas abaixo são ilustrativas.",
        "",
        "## Premissas",
        "",
        "| Premissa | Valor |",
        "|---|---:|",
        f"| Horizonte do valor em risco | {p.horizonte_meses} meses de mensalidade |",
        f"| Taxa de sucesso da retenção | {_dec(100 * p.taxa_sucesso, 0)}% |",
        f"| Custo de cada ação de retenção | {_usd(p.custo_acao)} |",
        "",
        "## Calibração das probabilidades (teste)",
        "",
        f"Churn real no teste: {_dec(100 * c['churn_real_teste'], 1)}%.",
        "",
        "| Modelo | Prob. média prevista (bruta) | Prob. média (calibrada) "
        "| Brier (bruto) | Brier (calibrado) |",
        "|---|---:|---:|---:|---:|",
        f"| XGBoost | {_dec(100 * c['prob_media_xgb_bruta'], 1)}% "
        f"| {_dec(100 * c['prob_media_xgb_calibrada'], 1)}% "
        f"| {_dec(c['brier_xgb_bruto'])} | {_dec(c['brier_xgb_calibrado'])} |",
        f"| Regressão Logística | — | — "
        f"| {_dec(c['brier_log_bruta'])} | {_dec(c['brier_log_calibrada'])} |",
        "",
        "## Estratégias de retenção (teste)",
        "",
        "| Estratégia | Contatados | Cancelamentos alcançados | Benefício total "
        "| Por 1.000 clientes |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in neg.estrategias.to_dict("records"):
        linhas.append(
            f"| {r['estrategia']} | {_int(r['contatados'])} "
            f"| {_int(r['cancelamentos_alcancados'])} | {_usd(r['beneficio_total'])} "
            f"| {_usd(r['beneficio_por_1000'])} |"
        )
    linhas += [
        "",
        "## Sensibilidade ao custo da ação "
        f"(taxa de sucesso {_dec(100 * p.taxa_sucesso, 0)}%)",
        "",
        "Benefício líquido por 1.000 clientes.",
        "",
        "| Custo da ação | Contatados (modelo) | Modelo calibrado "
        "| Regra antiga (≥ 0,5) | Contatar todos |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in neg.sens_custo.to_dict("records"):
        linhas.append(
            f"| {_usd(r['custo_acao'])} | {_int(r['contatados_modelo'])} "
            f"| {_usd(r['modelo'])} | {_usd(r['regra_antiga'])} | {_usd(r['todos'])} |"
        )
    linhas += [
        "",
        f"## Sensibilidade à taxa de sucesso (custo da ação {_usd(p.custo_acao)})",
        "",
        "| Taxa de sucesso | Contatados (modelo) | Modelo calibrado "
        "| Regra antiga (≥ 0,5) | Contatar todos |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in neg.sens_taxa.to_dict("records"):
        linhas.append(
            f"| {_dec(100 * r['taxa_sucesso'], 0)}% | {_int(r['contatados_modelo'])} "
            f"| {_usd(r['modelo'])} | {_usd(r['regra_antiga'])} | {_usd(r['todos'])} |"
        )
    linhas.append("")
    return "\n".join(linhas)
