"""Ponto de entrada: roda regressão e classificação. Uso: python main.py"""

from scripts.train_classification import main as rodar_classificacao
from scripts.train_regression import main as rodar_regressao


def main() -> None:
    rodar_regressao()
    rodar_classificacao()


if __name__ == "__main__":
    main()
