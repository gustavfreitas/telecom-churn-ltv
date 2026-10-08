# Pré-processamento reutilizável pelos dois problemas.

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def montar_preprocessador(
    cols_num: list[str], cols_cat: list[str], escalar: bool
) -> ColumnTransformer:
    #One-Hot nas categóricas; StandardScaler opcional nas numéricas.
    #As colunas restantes (binárias 0/1) passam sem alteração.
    num_transf = StandardScaler() if escalar else "passthrough"
    prep = ColumnTransformer(
        [
            ("num", num_transf, cols_num),
            (
                "cat",
                OneHotEncoder(
                    drop="first", handle_unknown="ignore", sparse_output=False
                ),
                cols_cat,
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
    return prep.set_output(transform="pandas")
