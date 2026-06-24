"""Loader de Mu-SHROOM-ES (SemEval-2025 Task 3, Vázquez et al. 2025).

Mu-SHROOM anota alucinaciones a nivel de span de carácter dentro de respuestas
generadas. Sus generaciones provienen de modelos ajenos a este trabajo (Qwen2-7B,
Llama-3-8B, Iker-Neurona), de modo que el uso es de puro transfer: se reutilizan
prompt, respuesta y etiquetas, y los features se extraen corriendo nuestros
modelos (Llama 3.2 3B, Gemma 3 4B) sobre (prompt, generación) por teacher forcing.
Es un conjunto de prueba zero-shot (clasificadores entrenados en MUCH-ES, sin
reajuste sobre estas muestras), según el Cap. 5 §5.6.

Solo se usan los splits etiquetados (validation 50 + test 152 = 202). Etiqueta a
nivel de muestra: alucinada (1) si hay al menos un span en `hard_labels`. Las
respuestas son largas (~42 palabras), el escenario de respuesta extensa donde los
métodos topológicos tienen su mejor caso. Clase mayoritariamente positiva (~88%).
"""

import ast
import os

import pandas as pd

from .. import config

from .base import Sample

_REPO = "Helsinki-NLP/mu-shroom"
_LABELED_SPLITS = ("validation", "test")


def _aslist(v):
    if isinstance(v, (list, tuple)):
        return list(v)
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except Exception:  # noqa: BLE001
            return []
    return []


def _cache_path():
    return config.BENCHMARKS_DIR / "mushroom" / "es.parquet"


def _build_cache() -> None:
    """Vuelca los splits etiquetados de la config `es` a un parquet. El texto es
    independiente del modelo (lo generó un modelo ajeno); la dependencia del modelo
    aparece solo en la extracción posterior."""
    from datasets import load_dataset

    tok = config.load_env_var("HF_TOKEN")
    if tok:
        os.environ["HF_TOKEN"] = tok

    ds = load_dataset(_REPO, "es")
    recs = []
    for split in _LABELED_SPLITS:
        for i, r in enumerate(ds[split]):
            hard = _aslist(r["hard_labels"])
            rid = r["id"] if r["id"] not in (None, "None", "") else f"{split}-{i:03d}"
            recs.append(
                {
                    "id": str(rid),
                    "prompt": str(r["model_input"]),
                    "respuesta": str(r["model_output_text"]).strip(),
                    "label": 1 if hard else 0,
                    "split": split,
                    "gen_model": str(r["model_id"]),
                }
            )
    out_dir = config.BENCHMARKS_DIR / "mushroom"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(recs).to_parquet(_cache_path())


def load_mushroom_es(max_samples: int | None = None) -> list[Sample]:
    """Muestras etiquetadas de Mu-SHROOM-ES (validation + test). El texto no
    depende de nuestro modelo; la partición P/R usa plantilla de chat plana."""
    path = _cache_path()
    if not path.exists():
        _build_cache()
    df = pd.read_parquet(path)
    if max_samples is not None:
        df = df.iloc[:max_samples]

    samples = []
    for _, r in df.iterrows():
        lab = int(r["label"])
        samples.append(
            Sample(
                id=f"mushroom-es-{r['id']}",
                prompt=str(r["prompt"]),
                respuesta=str(r["respuesta"]),
                label=lab,
                lang="es",
                source=f"{r['gen_model']}:{r['split']}",
                meta={"chat": "plain", "split": str(r["split"])},
            )
        )
    return samples
