"""21_pool.py [much|all] - agregación de scores por idioma (etapa SIN modelo).

Las features de LapEigvals y SEPs tienen dimensión distinta entre modelos, así que
la cifra de detección por idioma se obtiene agregando los *scores* fuera de fold de
los dos modelos de MUCH (Llama 3.2 3B y Gemma 3 4B), con su intervalo de confianza
por bootstrap. Es la cifra honesta en MUCH, de clase positiva escasa (~4%), donde la
media por fold con 2-3 positivos es muy ruidosa.

Escribe data/results/metrics/pooled.parquet y la imprime.

    .venv/bin/python code/scripts/21_pool.py much
"""

import sys

import pandas as pd

from tfm_hallu import config
from tfm_hallu.evaluation.pooling import pooled_metrics

METHODS = ("toha", "lapeigvals", "halluzig", "seps")


def main(which: str = "all") -> int:
    rows = []
    # Grupos a agregar: cada uno junta los dos modelos de MUCH para un benchmark.
    groups = []
    if which in ("much", "all"):
        groups += [f"much-{lang}" for lang in ("es", "en")]
    if which in ("mushroom", "all"):
        # Mu-SHROOM-ES: CV interna (techo en respuestas largas) y zero-shot desde MUCH-ES.
        groups += ["mushroom-es", "mushroom-es-zeroshot"]

    for bench in groups:
        entries = [(m, bench) for m in config.MUCH_MODELS]
        for method in METHODS:
            try:
                r = pooled_metrics(entries, method)
            except FileNotFoundError:
                continue
            r["group"] = bench
            rows.append(r)

    if not rows:
        print("No hay scores que agregar todavía.")
        return 1

    df = pd.DataFrame(rows)
    out = config.RESULTS_DIR / "metrics" / "pooled.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["entries"]).to_parquet(out, index=False)

    for group in df["group"].unique():
        sub = df[df["group"] == group]
        n = int(sub["n"].iloc[0])
        npos = int(sub["n_pos"].iloc[0])
        print(f"\n=== {group}  (n={n}, positivos={npos}, base rate={100*npos/n:.1f}%) ===")
        print(f"  {'método':12} {'AUROC [IC95]':24} {'AUPRC [IC95]':24}")
        for _, r in sub.iterrows():
            au = f"{r['auroc']:.3f} [{r['auroc_lo']:.3f},{r['auroc_hi']:.3f}]"
            ap = f"{r['auprc']:.3f} [{r['auprc_lo']:.3f},{r['auprc_hi']:.3f}]"
            print(f"  {r['method']:12} {au:24} {ap:24}")
    print(f"\nGuardado en {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "much"))
