"""Utilitários compartilhados: salvar figuras e métricas."""

import json

import matplotlib.pyplot as plt

from src.config import FIGURAS_DIR, METRICAS_PATH


def salvar_figura(fig: plt.Figure, nome: str) -> None:
    """Salva a figura em reports/figures/ e a fecha (sem plt.show())."""
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURAS_DIR / nome, dpi=150, bbox_inches="tight")
    plt.close(fig)


def salvar_metricas(secao: str, metricas: dict) -> None:
    """Atualiza reports/metrics.json com as métricas de uma seção."""
    METRICAS_PATH.parent.mkdir(parents=True, exist_ok=True)
    atual = {}
    if METRICAS_PATH.exists():
        atual = json.loads(METRICAS_PATH.read_text(encoding="utf-8"))
    atual[secao] = metricas
    METRICAS_PATH.write_text(
        json.dumps(atual, indent=2, ensure_ascii=False), encoding="utf-8"
    )
