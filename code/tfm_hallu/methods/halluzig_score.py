"""HalluZig: clasificador (etapa SIN modelo).

Siguiendo a Samaga et al. (2026) y el Cap. 5 §5.1.4: las features topológicas
(imagen de persistencia + entropía + curva de Betti, en H0 y H1) alimentan un
Random Forest de 200 árboles, elegido por su robustez frente al sobreajuste con
conjuntos de tamaño moderado y por permitir analizar la importancia de features.
Compatible con la interfaz `make_estimator(n_train)` del runner.
"""

from sklearn.ensemble import RandomForestClassifier

from .. import config


def make_halluzig_estimator(n_train: int | None = None, seed: int = config.SEED):
    return RandomForestClassifier(
        n_estimators=config.HALLUZIG_RF_TREES,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
