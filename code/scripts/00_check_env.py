"""00_check_env.py - Fase 0 del pipeline del Capitulo 6: verificacion del entorno.

Comprueba que las dependencias sensibles instalan y FUNCIONAN en Python 3.14
sobre macOS ARM, no solo que importan. Cubre los pasos F0.1-F0.4 del plan
(docs/plan_implementacion_cap6.md):

  F0.1  pyarrow      -> round-trip DataFrame <-> .parquet identico
  F0.2  zigzag       -> barcode zigzag de un ejemplo minimo comprobado a mano
  F0.3  bert-score   -> P/R/F1 sobre un par de frases
  F0.4  datasets     -> la libreria construye un Dataset en Py3.14

Backend de zigzag verificado: Dionysus 2 (red de seguridad para HalluZig,
Fase 3). FastZigzag no es un paquete de PyPI (es C++ con bindings que hay que
compilar), asi que su uso se evaluara en la Fase 3 comparando tiempos reales;
GUDHI no expone persistencia zigzag en su API de Python (solo estandar).

Uso:
    .venv/bin/python code/scripts/00_check_env.py

Codigo de salida 0 si todas las comprobaciones esenciales pasan, 1 en caso
contrario.
"""

import importlib.metadata as ilm
import math
import os
import sys
import tempfile

# Modelo pequeno y bien soportado para la prueba funcional de bert-score.
# El modelo de produccion para SelfCheckGPT (multilingue, EN+ES) se decide en
# la Fase 3; aqui solo se verifica que la libreria corre de punta a punta.
BERTSCORE_PROBE_MODEL = "distilbert-base-uncased"


def _ver(pkg: str) -> str:
    try:
        return ilm.version(pkg)
    except Exception:  # noqa: BLE001
        return "?"


def check_core():
    """Stack ya validado (smoke tests): solo se reporta version + MPS."""
    import torch

    mps = torch.backends.mps.is_available()
    det = (
        f"torch {_ver('torch')}, transformers {_ver('transformers')}, "
        f"ripser {_ver('ripser')}, persim {_ver('persim')}, "
        f"scikit-learn {_ver('scikit-learn')}; MPS={'si' if mps else 'NO'}"
    )
    return mps, det


def check_pyarrow():
    """F0.1: round-trip DataFrame -> parquet -> DataFrame identico."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "label": [0, 1, 0],
            "score": [0.12, 0.98, 0.34],
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "probe.parquet")
        df.to_parquet(path, engine="pyarrow")
        back = pd.read_parquet(path, engine="pyarrow")
    ok = df.equals(back)
    return ok, f"pyarrow {_ver('pyarrow')}: round-trip {'identico' if ok else 'DIFIERE'}"


def check_zigzag():
    """F0.2: persistencia zigzag con Dionysus 2 sobre un ejemplo minimo.

    Ejemplo: dos vertices presentes siempre; la arista [0,1] entra en t=1 y
    sale en t=2. Razonamiento a mano sobre H0 (componentes conexas):
      - clase esencial siempre viva                       -> [0, inf)
      - 2a componente que muere al conectar en t=1        -> [0, 1)
      - componente que renace al quitar la arista en t=2  -> [2, inf)
    Es decir: 3 barras en H0, una finita [0,1] y dos infinitas.
    """
    import dionysus as d

    f = d.Filtration([[0], [1], [0, 1]])
    times = [[0.0], [0.0], [1.0, 2.0]]
    _, dgms, _ = d.zigzag_homology_persistence(f, times)
    h0 = list(dgms[0]) if len(dgms) > 0 else []

    finitas = [(p.birth, p.death) for p in h0 if not math.isinf(p.death)]
    infinitas = [(p.birth, p.death) for p in h0 if math.isinf(p.death)]
    ok = (
        len(h0) == 3
        and len(infinitas) == 2
        and len(finitas) == 1
        and abs(finitas[0][0] - 0.0) < 1e-9
        and abs(finitas[0][1] - 1.0) < 1e-9
    )
    det = (
        f"Dionysus {_ver('dionysus')}: H0 = "
        f"{len(finitas)} finita(s) {finitas} + {len(infinitas)} infinita(s) "
        f"{'(correcto a mano)' if ok else '(INESPERADO)'}"
    )
    return ok, det


def check_bertscore():
    """F0.3: bert-score produce P/R/F1 sobre un par de frases."""
    from bert_score import score

    cands = ["the cat sat on the mat"]
    refs = ["a cat is sitting on the mat"]
    P, R, F = score(
        cands,
        refs,
        model_type=BERTSCORE_PROBE_MODEL,
        num_layers=5,
        verbose=False,
        lang="en",
    )
    f1 = float(F[0])
    ok = 0.0 < f1 < 1.0
    return ok, (
        f"bert-score {_ver('bert-score')} (modelo de prueba {BERTSCORE_PROBE_MODEL}): "
        f"F1={f1:.3f} {'OK' if ok else 'FUERA DE RANGO'}"
    )


def check_datasets():
    """F0.4: la libreria datasets construye un Dataset en Py3.14.

    Se verifica que la libreria funciona (importa pyarrow por debajo) sin
    depender de la red. La carga desde el Hub de MUCH y Mu-SHROOM se prueba
    cuando toque su descarga (Fase 4).
    """
    from datasets import Dataset

    ds = Dataset.from_dict({"q": ["a", "b"], "label": [0, 1]})
    ok = len(ds) == 2 and ds[0]["q"] == "a"
    return ok, (
        f"datasets {_ver('datasets')}: Dataset.from_dict ok "
        f"(carga desde el Hub se valida en Fase 4)"
    )


# (nombre, funcion, esencial)
CHECKS = [
    ("core (torch/MPS)", check_core, True),
    ("F0.1 pyarrow", check_pyarrow, True),
    ("F0.2 zigzag (Dionysus 2)", check_zigzag, True),
    ("F0.3 bert-score", check_bertscore, True),
    ("F0.4 datasets", check_datasets, True),
]


def main() -> int:
    print(f"=== Fase 0: verificacion del entorno (Python {sys.version.split()[0]}) ===\n")
    resultados = []
    for nombre, fn, esencial in CHECKS:
        try:
            ok, det = fn()
        except Exception as e:  # noqa: BLE001
            ok, det = False, f"EXCEPCION {type(e).__name__}: {e}"
        resultados.append((nombre, ok, esencial))
        marca = "OK  " if ok else "FALLO"
        print(f"  [{marca}] {nombre}\n          {det}\n")

    esenciales_ok = all(ok for _, ok, es in resultados if es)
    print("=" * 60)
    print(f"Esenciales: {'TODAS OK' if esenciales_ok else 'HAY FALLOS'}")
    return 0 if esenciales_ok else 1


if __name__ == "__main__":
    sys.exit(main())
