"""42_master_table.py - reanalisis SIN modelo: tabla maestra de AUROC del Cap. 6.

Lee las tablas ya versionadas en data/results/metrics/ (no necesita modelos ni
.pt) y construye la tabla maestra de AUROC media (sobre los folds de CV) por
benchmark x modelo x metodo, mas un resumen de eficiencia. Pensado para que
Alvaro reanalice los resultados de Alejandro sin re-extraer nada.

    .venv/bin/python code/scripts/42_master_table.py

Fuentes:
  - detection.parquet : AUROC/AUPRC/F1/FPR95 por (model, benchmark, method, fold)
  - efficiency.parquet: seconds_per_sample y n_forward por (model, benchmark, method)
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "data" / "results" / "metrics"

# Orden de presentacion (familias: topologicas, espectral, estados ocultos, caja negra)
METHOD_ORDER = ["toha", "halluzig", "lapeigvals", "seps", "selfcheckgpt", "length"]


def _ordered(cols):
    """Ordena los metodos segun METHOD_ORDER, dejando los desconocidos al final."""
    known = [m for m in METHOD_ORDER if m in cols]
    rest = [c for c in cols if c not in METHOD_ORDER]
    return known + rest


def main() -> int:
    det_path = METRICS / "detection.parquet"
    if not det_path.exists():
        print(f"No encuentro {det_path}. ¿Estás en la raíz del repo?")
        return 1

    det = pd.read_parquet(det_path)

    # AUROC media sobre los folds de cada (benchmark, model, method).
    g = (
        det.groupby(["benchmark", "model", "method"])["auroc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # Tabla maestra: filas (benchmark, model), columnas metodo, valor AUROC medio.
    piv = g.pivot_table(index=["benchmark", "model"], columns="method", values="mean")
    piv = piv[_ordered(list(piv.columns))]

    pd.set_option("display.width", 160)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")

    print("\n" + "=" * 72)
    print("TABLA MAESTRA DE AUROC (media sobre folds de CV)")
    print("filas: benchmark x modelo   |   columnas: metodo")
    print("=" * 72)
    print(piv.to_string(na_rep="  -  "))

    # Mejor metodo por fila (orienta el mapa de regimenes).
    print("\n" + "-" * 72)
    print("Mejor metodo por benchmark x modelo:")
    print("-" * 72)
    best = piv.idxmax(axis=1)
    bestval = piv.max(axis=1)
    for idx, m in best.items():
        bench, model = idx
        print(f"  {bench:<22} {model:<14} -> {m:<12} ({bestval[idx]:.3f})")

    # Eficiencia.
    eff_path = METRICS / "efficiency.parquet"
    if eff_path.exists():
        eff = pd.read_parquet(eff_path)
        cost = (
            eff.groupby("method")[["seconds_per_sample", "n_forward"]]
            .mean()
            .reindex(_ordered(eff["method"].unique()))
        )
        print("\n" + "-" * 72)
        print("Coste medio por metodo (s/muestra y nº de forward passes):")
        print("-" * 72)
        print(cost.to_string(na_rep="  -  "))

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
