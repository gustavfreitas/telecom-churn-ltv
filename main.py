from src.data import carregar_e_limpar
from src.regressao import executar_regressao
from src.classificacao import executar_classificacao
from src.config import DATA_PATH


def main() -> None:
    df = carregar_e_limpar(DATA_PATH)
    executar_regressao(df)
    executar_classificacao(df)


if __name__ == "__main__":
    main()