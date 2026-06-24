"""Agregación de scores a nivel de muestra (Fase 4).

Las features de LapEigvals (L·H·k) y SEPs (L×d_model) tienen dimensión distinta
entre modelos, de modo que no se pueden concatenar a nivel de feature para un
clasificador conjunto. La agregación «por idioma, varios modelos» del Cap. 6 se
hace, por tanto, a nivel de *score*: cada modelo produce sus scores fuera de fold
(`scores/<modelo>_<benchmark>_<metodo>.parquet`), y aquí se concatenan score y
etiqueta de las muestras de varios (modelo, benchmark) para una única cifra de
detección con intervalo de confianza por bootstrap.

Es la cifra honesta en MUCH (clase positiva ~4%): la media por fold es muy
ruidosa con 2-3 positivos por pliegue, mientras que el AUROC/AUPRC agrupado sobre
todas las muestras fuera de fold, con su CI, resume mejor la capacidad real.
"""

import pandas as pd

from .. import config
from .metrics import bootstrap_ci, detection_metrics


def _scores_path(model_key, bench_key, method):
    return config.RESULTS_DIR / "scores" / f"{model_key}_{bench_key}_{method}.parquet"


def pool_scores(entries, method):
    """Concatena (score, label) de varias combinaciones (modelo, benchmark) para
    un método. `entries` es una lista de (model_key, bench_key). Devuelve un
    DataFrame con columnas model, benchmark, sample_id, score, label."""
    frames = []
    for model_key, bench_key in entries:
        df = pd.read_parquet(_scores_path(model_key, bench_key, method))
        df = df.assign(model=model_key, benchmark=bench_key)
        frames.append(df[["model", "benchmark", "sample_id", "score", "label"]])
    return pd.concat(frames, ignore_index=True)


def pooled_metrics(entries, method, metrics=("auroc", "auprc", "f1", "fpr95")):
    """Métricas agrupadas + CIs bootstrap sobre los scores fuera de fold de varias
    combinaciones (modelo, benchmark). Devuelve un dict con n, n_pos, las métricas
    puntuales y, por métrica, el intervalo [lo, hi]."""
    df = pool_scores(entries, method)
    y = df["label"].to_numpy()
    s = df["score"].to_numpy()
    valid = ~pd.isna(s)
    y, s = y[valid], s[valid]
    point = detection_metrics(y, s)
    ci = bootstrap_ci(y, s, metrics=metrics)
    return {
        "method": method,
        "entries": list(entries),
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        **{m: point[m] for m in metrics},
        **{f"{m}_lo": ci[m]["lo"] for m in metrics},
        **{f"{m}_hi": ci[m]["hi"] for m in metrics},
    }
