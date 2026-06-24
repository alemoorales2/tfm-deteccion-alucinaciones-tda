"""Loader del HaluEval-QA generado por el modelo.

Lee el dataset que produce `05_generate.py` (una respuesta generada por ítem,
etiquetada con el híbrido exact-match + juez Gemini) y lo expone como `Sample`.
Es específico de cada modelo (cada uno genera sus propias alucinaciones).

La partición P/R se hace con plantilla de chat (meta["chat"]=True): P es el
prompt closed-book tal como lo vio el modelo al generar, R la respuesta generada.
"""

import pandas as pd

from .. import config
from .base import Sample


def _path(model_key: str):
    return config.BENCHMARKS_DIR / "halueval-qa-gen" / f"{model_key}.parquet"


def load_halueval_qa_generated(model_key: str, max_samples: int | None = None) -> list[Sample]:
    df = pd.read_parquet(_path(model_key))
    if max_samples is not None:
        df = df.iloc[:max_samples]
    samples = []
    for _, r in df.iterrows():
        lab = int(r["label"])
        samples.append(
            Sample(
                id=f"halueval-qa-gen-{model_key}-{int(r['item']):05d}",
                prompt=str(r["question"]),       # P se construye con plantilla de chat en encode
                respuesta=str(r["generated"]),
                label=lab,
                lang="en",
                source="gen-hallucinated" if lab == 1 else "gen-factual",
                meta={"item": int(r["item"]), "chat": True},
            )
        )
    return samples
