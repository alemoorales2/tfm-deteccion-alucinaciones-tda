"""Loader de SQuAD para el control de reproducción de TOHA (Fase 4.9).

SQuAD (Rajpurkar et al. 2016) es comprensión lectora: cada ítem trae un párrafo
de contexto, una pregunta y una respuesta dorada extractiva (corta). Es uno de los
benchmarks donde TOHA reporta su mejor AUROC (0,87-0,96), precisamente porque el
prompt incluye un contexto P sustancioso del que la respuesta R puede «desviarse».
Se usa como control: si nuestra reimplementación de TOHA reproduce esos números
aquí, el bajo rendimiento en los benchmarks closed-book/abiertos (HaluEval, MUCH,
Mu-SHROOM) es un efecto de régimen (ausencia de contexto P), no un error.

El benchmark es open-book y generado por nuestro modelo: este responde con el
contexto en el prompt, y la respuesta se etiqueta sin juez (exact-match normalizado
contra cualquiera de las respuestas doradas; ver `05_generate_squad.py`). El campo
`prompt` almacena el mensaje de usuario exacto (contexto + pregunta) para que la
partición P/R de la extracción reproduzca lo que vio el modelo (modo chat «plain»).
"""

from collections import Counter

import pandas as pd

from .. import config
from ..labeling import normalize
from .base import Sample

SQUAD_INSTRUCTION = (
    "Answer the question based only on the context below, with a short, direct "
    "answer of a few words."
)


def squad_user_content(context: str, question: str) -> str:
    """Mensaje de usuario único (lo usan generación y extracción)."""
    return (
        f"{SQUAD_INSTRUCTION}\n\n"
        f"Context: {context.strip()}\n\n"
        f"Question: {question.strip()}"
    )


def _contig(a: list[str], b: list[str]) -> bool:
    """True si `a` es subsecuencia contigua de palabras de `b`."""
    return bool(a) and any(b[i : i + len(a)] == a for i in range(len(b) - len(a) + 1))


def _token_f1(g: list[str], t: list[str]) -> float:
    if not g or not t:
        return 0.0
    common = sum((Counter(g) & Counter(t)).values())
    if common == 0:
        return 0.0
    p, r = common / len(t), common / len(g)
    return 2 * p * r / (p + r)


def squad_is_correct(golds, gen: str) -> bool:
    """Etiquetado al estilo SQuAD: la respuesta generada es correcta si, frente a
    alguna respuesta dorada, hay igualdad normalizada, contención en cualquiera de
    los dos sentidos (cubre doradas que son enumeraciones de las que el modelo da un
    elemento válido) o F1 de tokens >= 0,5. Conservador hacia «correcta» para no
    inflar la clase alucinada con paráfrasis válidas."""
    t = normalize(gen)
    for gold in golds:
        g = normalize(gold)
        if not g:
            continue
        if g == t or _contig(g, t) or _contig(t, g) or _token_f1(g, t) >= 0.5:
            return True
    return False


def _path(model_key: str):
    return config.BENCHMARKS_DIR / "squad" / f"{model_key}.parquet"


def load_squad(model_key: str, max_samples: int | None = None) -> list[Sample]:
    """Muestras de SQuAD generadas y etiquetadas por `05_generate_squad.py`."""
    df = pd.read_parquet(_path(model_key))
    if max_samples is not None:
        df = df.iloc[:max_samples]
    samples = []
    for _, r in df.iterrows():
        lab = int(r["label"])
        samples.append(
            Sample(
                id=f"squad-{model_key}-{r['id']}",
                prompt=str(r["prompt"]),       # contexto + pregunta (se templará en encode)
                respuesta=str(r["generated"]),
                label=lab,
                lang="en",
                source="rc-hallucinated" if lab == 1 else "rc-factual",
                meta={"chat": "plain"},
            )
        )
    return samples
