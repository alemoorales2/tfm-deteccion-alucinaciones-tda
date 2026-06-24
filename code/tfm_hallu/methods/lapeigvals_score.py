"""LapEigvals: clasificador (etapa SIN modelo).

Siguiendo a Binkowski et al. (2025): el vector de autovalores L*H*k se proyecta
con PCA a 512 dimensiones y se clasifica con regresión logística. La memoria
§5.4 pide ajuste por validación cruzada interna, así que se usa
`LogisticRegressionCV` (ℓ2) en lugar de C fijo. `class_weight="balanced"` y
`max_iter=2000` como en el paper.
"""

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline

from .. import config


def make_lapeigvals_estimator(n_features, n_train, seed=config.SEED):
    n_pca = min(config.LAPEIGVALS_PCA, n_train - 1, n_features)
    return Pipeline(
        [
            ("pca", PCA(n_components=n_pca, random_state=seed)),
            (
                "lr",
                LogisticRegressionCV(
                    Cs=10,
                    cv=5,
                    penalty="l2",
                    scoring="roc_auc",
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]
    )
