#Executa classificação + interpretabilidade.
#Rode da raiz: python -m scripts.train_classification.

from src.classification import executar_classificacao
from src.data import carregar_e_limpar
from src.evaluation import salvar_metricas
from src.interpretability import churn_por_segmento, gerar_importancias, top_risco


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


if __name__ == "__main__":
    main()
