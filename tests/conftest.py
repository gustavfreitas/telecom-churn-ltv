#Configuração compartilhada dos testes.

import matplotlib
import pytest

matplotlib.use("Agg")  # backend sem janela: testes não abrem figuras

from src.data import carregar_e_limpar  # noqa: E402


@pytest.fixture(scope="session")
def df():
    #DataFrame limpo, carregado uma única vez para toda a sessão.
    return carregar_e_limpar()
