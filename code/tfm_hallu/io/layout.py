"""Layout de resultados en disco (diseño §6): .pt para tensores, .parquet para
tablas. Centraliza rutas y E/S para que scripts y análisis no las repitan.

    data/results/<modelo>/<benchmark>/
        meta.json            metadatos de la corrida
        samples.parquet      índice de muestras (esquema del diseño §6)
        toha_headdiv.pt      tensor [n, L, H]            (TOHA)
        lapeigvals_feats.pt  tensor [n, L, H, k]         (LapEigvals)
    data/results/scores/<modelo>_<benchmark>_<metodo>.parquet
    data/results/metrics/detection.parquet
"""

import json

import pandas as pd
import torch

from .. import config

_KEY = ["model", "benchmark", "method"]


def result_dir(model_key, bench_key, create=False):
    d = config.RESULTS_DIR / model_key / bench_key
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def save_extraction(model_key, bench_key, arrays, samples_df, meta):
    """Persiste la extracción. `arrays` es un dict nombre -> np.ndarray; cada
    uno se guarda como `<nombre>.pt`."""
    d = result_dir(model_key, bench_key, create=True)
    for name, arr in arrays.items():
        torch.save(torch.from_numpy(arr), d / f"{name}.pt")
    samples_df.to_parquet(d / "samples.parquet", index=False)
    (d / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def save_barcodes(model_key, bench_key, rows):
    """Persiste los barcodes zigzag de HalluZig por muestra en
    `halluzig_barcodes.parquet` (columnas sample_id, n_levels, h0, h1; h0/h1 son
    JSON de listas [birth, death]). Permite re-vectorizar HalluZig sin re-extraer."""
    d = result_dir(model_key, bench_key, create=True)
    pd.DataFrame(rows).to_parquet(d / "halluzig_barcodes.parquet", index=False)


def load_array(model_key, bench_key, name):
    return torch.load(result_dir(model_key, bench_key) / f"{name}.pt").numpy()


def load_samples(model_key, bench_key):
    return pd.read_parquet(result_dir(model_key, bench_key) / "samples.parquet")


def load_meta(model_key, bench_key):
    return json.loads(
        (result_dir(model_key, bench_key) / "meta.json").read_text(encoding="utf-8")
    )


def save_scores(model_key, bench_key, method, df):
    d = config.RESULTS_DIR / "scores"
    d.mkdir(parents=True, exist_ok=True)
    df.to_parquet(d / f"{model_key}_{bench_key}_{method}.parquet", index=False)


def append_metrics(rows):
    """Añade filas a metrics/detection.parquet, reemplazando las de la misma
    combinación (modelo, benchmark, método) si ya existían."""
    d = config.RESULTS_DIR / "metrics"
    d.mkdir(parents=True, exist_ok=True)
    path = d / "detection.parquet"
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_parquet(path)
        newkeys = set(map(tuple, new[_KEY].itertuples(index=False, name=None)))
        keep = old[~old[_KEY].apply(lambda r: tuple(r) in newkeys, axis=1)]
        combined = pd.concat([keep, new], ignore_index=True)
    else:
        combined = new
    combined.to_parquet(path, index=False)
