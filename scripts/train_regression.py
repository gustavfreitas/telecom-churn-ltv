"""Executa a etapa de regressão. Rode da raiz: python -m scripts.train_regression"""

from src.data import carregar_e_limpar, carregar_ids
from src.evaluation import salvar_metricas
from src.regression import auditar_cobranca, executar_regressao, salvar_auditoria


def main() -> None:
    df = carregar_e_limpar()
    metricas = executar_regressao(df)
    salvar_metricas("regressao", metricas)
    print(metricas)

    # Lista de clientes com cobrança fora do padrão (candidatos a auditoria)
    auditoria, resumo = auditar_cobranca(df, carregar_ids())
    salvar_auditoria(auditoria)
    salvar_metricas("auditoria_cobranca", resumo)
    print("\nAuditoria de cobrança:", resumo)
    print(auditoria.head(5).round(2))


if __name__ == "__main__":
    main()
