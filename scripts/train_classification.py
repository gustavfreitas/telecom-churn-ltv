"""Executa classificação + interpretabilidade.
Rode da raiz: python -m scripts.train_classification
"""

from src import config
from src.classification import executar_classificacao
from src.data import carregar_e_limpar, carregar_ids
from src.evaluation import salvar_metricas
from src.interpretability import (
    churn_por_segmento,
    gerar_importancias,
    motivos_por_cliente,
    top_risco,
)
from src.negocio import (
    executar_analise_negocio,
    metricas_negocio,
    montar_fila_retencao,
    renderizar_markdown,
    salvar_fila,
)


def main() -> None:
    df = carregar_e_limpar()
    res = executar_classificacao(df)
    salvar_metricas("classificacao", res.metricas)
    print(res.metricas)

    print(gerar_importancias(res).head(10).round(4))
    for coluna, serie in churn_por_segmento(
        df, ["Contract", "InternetService", "PaymentMethod"]
    ).items():
        print(f"\nChurn por {coluna}:\n{serie}")
    print("\nTop 10 clientes em risco:\n", top_risco(res))

    # Custo de negócio: calibra as probabilidades e compara estratégias de retenção
    neg = executar_analise_negocio(res)
    salvar_metricas("negocio", metricas_negocio(neg))
    config.RESULTADOS_NEGOCIO_PATH.write_text(
        renderizar_markdown(neg), encoding="utf-8"
    )
    print("\n", neg.estrategias.round(1).to_string(index=False))

    # Fila de retenção com os motivos de cada cliente (SHAP nativo do XGBoost)
    motivos = motivos_por_cliente(res)
    salvar_fila(montar_fila_retencao(res, neg, carregar_ids(), motivos))
    print(f"\nFila de retenção salva em {config.FILA_PATH}")


if __name__ == "__main__":
    main()
