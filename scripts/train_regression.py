#Executa a etapa de regressão. Rode da raiz: python -m scripts.train_regression.

from src.data import carregar_e_limpar
from src.evaluation import salvar_metricas
from src.regression import executar_regressao


def main() -> None:
    df = carregar_e_limpar()
    metricas = executar_regressao(df)
    salvar_metricas("regressao", metricas)
    print(metricas)


if __name__ == "__main__":
    main()
