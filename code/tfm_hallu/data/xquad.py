"""Loader de XQuAD para la comparación cross-language CON contexto (Fase 4).

XQuAD (Artetxe et al. 2020) es comprensión lectora paralela en once idiomas: las
mismas preguntas y párrafos de contexto, traducidos profesionalmente. Aquí se usa la
pareja inglés/español para la pregunta cross-language en el régimen donde los métodos
topológicos rinden (contexto P sustancioso en el prompt, como en SQuAD). Al ser las
preguntas idénticas salvo el idioma, la comparación EN vs ES queda controlada: cambia
el idioma y nada más.

Mismo patrón que `squad.py`: open-book y generado por el propio modelo (responde con
el contexto en el prompt), etiqueta híbrida exact-match/F1 + juez Gemini para las
ambiguas (ver `05_generate_xquad.py`). El campo `prompt` guarda el mensaje de usuario
exacto (contexto + pregunta) para que la partición P/R reproduzca lo que vio el modelo
(modo chat «plain»). Reutiliza las funciones de emparejamiento de `squad.py` y, en
español, la normalización consciente de acentos/artículos de `labeling.normalize_es`.
"""

import pandas as pd

from .. import config
from ..labeling import normalize, normalize_es
from .base import Sample
from .squad import _contig, _token_f1

XQUAD_INSTRUCTION_EN = (
    "Answer the question based only on the context below, with a short, direct "
    "answer of a few words."
)
XQUAD_INSTRUCTION_ES = (
    "Responde a la pregunta basándote únicamente en el contexto siguiente, con una "
    "respuesta corta y directa de pocas palabras."
)


def xquad_user_content(context: str, question: str, lang: str) -> str:
    """Mensaje de usuario único (lo usan generación y extracción), en el idioma del
    benchmark para que el modelo responda en ese idioma."""
    if lang == "es":
        return (
            f"{XQUAD_INSTRUCTION_ES}\n\n"
            f"Contexto: {context.strip()}\n\n"
            f"Pregunta: {question.strip()}"
        )
    return (
        f"{XQUAD_INSTRUCTION_EN}\n\n"
        f"Context: {context.strip()}\n\n"
        f"Question: {question.strip()}"
    )


def xquad_is_correct(golds, gen: str, lang: str) -> bool:
    """Etiquetado al estilo SQuAD (igualdad / contención bidireccional / F1>=0,5
    contra alguna respuesta dorada), con la normalización propia del idioma: en
    español se quitan acentos y artículos españoles (`normalize_es`); en inglés, la
    `normalize` anglocéntrica. Conservador hacia «correcta» para no inflar la clase
    alucinada con paráfrasis válidas."""
    norm = normalize_es if lang == "es" else normalize
    t = norm(gen)
    for gold in golds:
        g = norm(gold)
        if not g:
            continue
        if g == t or _contig(g, t) or _contig(t, g) or _token_f1(g, t) >= 0.5:
            return True
    return False


def _path(model_key: str, lang: str):
    return config.BENCHMARKS_DIR / f"xquad-{lang}" / f"{model_key}.parquet"


def load_xquad(model_key: str, lang: str, max_samples: int | None = None) -> list[Sample]:
    """Muestras de XQuAD generadas y etiquetadas por `05_generate_xquad.py`."""
    df = pd.read_parquet(_path(model_key, lang))
    if max_samples is not None:
        df = df.iloc[:max_samples]
    samples = []
    for _, r in df.iterrows():
        lab = int(r["label"])
        samples.append(
            Sample(
                id=f"xquad-{lang}-{model_key}-{r['id']}",
                prompt=str(r["prompt"]),       # contexto + pregunta (se templará en encode)
                respuesta=str(r["generated"]),
                label=lab,
                lang=lang,
                source="rc-hallucinated" if lab == 1 else "rc-factual",
                meta={"chat": "plain"},
            )
        )
    return samples
