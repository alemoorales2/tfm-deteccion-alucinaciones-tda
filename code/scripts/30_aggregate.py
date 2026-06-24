"""30_aggregate.py - deriva metricas internas SIN modelo desde los artefactos persistidos.

Recorre data/results/<modelo>/<benchmark>/ y produce dos tablas en data/results/metrics/:

  - topological.parquet: por muestra (model, benchmark, sample_id, label, lang) con las
    metricas topologicas internas (Cap. 5 §5.5.2) derivadas de los .pt ya guardados:
      * entropia de persistencia en H0 y H1            <- halluzig_feats (escalares directos)
      * numeros de Betti persistentes b0, b1           <- maximo de la curva de Betti H0/H1
      * suma de persistencia (total persistence) H0/H1 <- area bajo la curva de Betti
      * divergencia TOHA media (MTop-Div por muestra)  <- media de toha_headdiv sobre (L,H)
    No son puntuaciones de decision; se reportan por grupo factual/alucinado (figura 6).

  - efficiency.parquet: por (model, benchmark, method) con el coste de computo (figura 7):
      * seconds_per_sample (de meta.json; la pasada de extraccion es compartida por los
        cuatro metodos de internos, que derivan de un unico forward)
      * n_forward: 1 para los metodos de internos, N+1=6 para SelfCheckGPT
      * n_truncated (diagnostico)

    .venv/bin/python code/scripts/30_aggregate.py

Layout de halluzig_feats (D=902, ver methods/halluzig.py): por dimension homologica
(0 y 1) se concatenan imagen de persistencia (pi_res^2), entropia (1) y curva de Betti
(betti_points). Con pi_res=20 y betti_points=50: cada bloque mide 451, total 902.
"""

import json
import sys

import numpy as np
import pandas as pd
import torch

from tfm_hallu import config

INTERNAL_METHODS = ("toha", "lapeigvals", "halluzig", "seps")
N_FORWARD = {"toha": 1, "lapeigvals": 1, "halluzig": 1, "seps": 1, "selfcheckgpt": 6}


def _halluzig_slices(pi_res: int, betti_points: int):
    """Indices (entropia, betti) por dimension homologica en el vector D=902."""
    block = pi_res * pi_res + 1 + betti_points  # 451 por defecto
    pi = pi_res * pi_res
    h0 = {"ent": pi, "betti": slice(pi + 1, block)}
    h1 = {"ent": block + pi, "betti": slice(block + pi + 1, 2 * block)}
    return h0, h1


def _topological_rows(model: str, bench: str, d: "Path") -> list[dict]:  # noqa: F821
    meta = json.load(open(d / "meta.json"))
    samples = pd.read_parquet(d / "samples.parquet")
    feats = torch.load(d / "halluzig_feats.pt", map_location="cpu").numpy()
    toha = torch.load(d / "toha_headdiv.pt", map_location="cpu").numpy()  # [n,L,H]
    pi_res = int(meta.get("halluzig_pi_res", config.HALLUZIG_PI_RES))
    bp = int(meta.get("halluzig_betti_points", config.HALLUZIG_BETTI_POINTS))
    h0, h1 = _halluzig_slices(pi_res, bp)

    n = len(samples)
    toha_mean = toha.reshape(n, -1).mean(axis=1)  # divergencia media sobre cabezas
    rows = []
    for i, r in enumerate(samples.itertuples(index=False)):
        b0 = feats[i, h0["betti"]]
        b1 = feats[i, h1["betti"]]
        rows.append({
            "model": model, "benchmark": bench,
            "sample_id": r.sample_id, "label": int(r.label), "lang": r.lang,
            "ent_h0": float(feats[i, h0["ent"]]), "ent_h1": float(feats[i, h1["ent"]]),
            "betti0": float(b0.max()), "betti1": float(b1.max()),
            "totpers_h0": float(b0.mean()), "totpers_h1": float(b1.mean()),
            "toha_div": float(toha_mean[i]),
        })
    return rows


def _efficiency_rows(model: str, bench: str, d: "Path") -> list[dict]:  # noqa: F821
    meta = json.load(open(d / "meta.json"))
    sps = float(meta.get("seconds_per_sample", float("nan")))
    ntr = int(meta.get("n_truncated", 0))
    rows = []
    for m in INTERNAL_METHODS:
        rows.append({"model": model, "benchmark": bench, "method": m,
                     "seconds_per_sample": sps, "n_forward": N_FORWARD[m], "n_truncated": ntr})
    # SelfCheckGPT: solo donde existan sus generaciones (subconjunto acotado, ~Phi)
    if (d / "selfcheck_gens.parquet").exists():
        rows.append({"model": model, "benchmark": bench, "method": "selfcheckgpt",
                     "seconds_per_sample": float("nan"), "n_forward": N_FORWARD["selfcheckgpt"],
                     "n_truncated": ntr})
    return rows


def main() -> int:
    res = config.RESULTS_DIR
    topo, eff = [], []
    for model_dir in sorted(p for p in res.iterdir() if p.is_dir() and p.name in config.MODELS):
        for d in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            if not (d / "meta.json").exists():
                continue
            bench = d.name
            if (d / "halluzig_feats.pt").exists() and (d / "toha_headdiv.pt").exists():
                topo.extend(_topological_rows(model_dir.name, bench, d))
            eff.extend(_efficiency_rows(model_dir.name, bench, d))

    out = config.RESULTS_DIR / "metrics"
    out.mkdir(parents=True, exist_ok=True)
    df_t = pd.DataFrame(topo)
    df_e = pd.DataFrame(eff)
    df_t.to_parquet(out / "topological.parquet", index=False)
    df_e.to_parquet(out / "efficiency.parquet", index=False)
    print(f"topological.parquet: {len(df_t)} muestras, "
          f"{df_t[['model','benchmark']].drop_duplicates().shape[0]} celdas modelo x benchmark")
    print(df_t.groupby(["model", "benchmark"]).size().to_string())
    print(f"\nefficiency.parquet: {len(df_e)} filas")
    print(df_e.head(10).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
