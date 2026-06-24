"""Loader de MEMERAG-es para el experimento RAG (Fase 5).

MEMERAG (Bldecic et al., Amazon Science, ACL 2025) es un benchmark de meta-evaluación
RAG multilingüe con preguntas nativas (sobre MIRACL) y, por pregunta, los pasajes
recuperados (top-5) y una respuesta generada por modelos grandes, anotada FRASE A
FRASE con su fidelidad al contexto («Supported» / «Not Supported» / «Challenging to
determine») por anotadores humanos.

Aquí se usa el split español en el régimen de DETECCIÓN SOBRE RESPUESTAS DADAS (estilo
RAGTruth, el benchmark con el que TOHA y HalluZig se evalúan en sus papers): la respuesta
NO la genera nuestro modelo, sino que se le pasa por *teacher forcing* junto al contexto
para extraer sus internos. La etiqueta es humana (no de juez), lo que da una verdad-terreno
limpia y un base rate sano (~44% en `memerag`, ~22% en la variante de voto de mayoría),
frente al ~5% que produce la auto-generación (el modelo se abstiene). El precio, asumido y
documentado, es que se detecta la (in)fidelidad de una respuesta ajena, no la propia.

Etiqueta a nivel respuesta: clase 1 (alucinada/infiel) si ALGUNA frase es «Not Supported»;
«Challenging to determine» se trata como clase 0 (no es una afirmación falsa). El campo
`prompt` guarda el mensaje de usuario exacto (instrucción + pasajes + pregunta) para que la
partición P/R de la extracción reproduzca el contexto sobre el que se anotó.

Requiere el repo clonado en data/benchmarks/memerag-repo/ (gitignored):
    git clone --depth 1 https://github.com/amazon-science/MEMERAG.git \\
        data/benchmarks/memerag-repo
"""

import json

from .. import config
from .base import Sample

# Instrucción en el idioma del benchmark (prompt monolingüe, para que la comparación
# cross-language no quede confundida por mezclar idiomas en la P, igual que en XQuAD).
MEMERAG_INSTRUCTION = {
    "es": ("Responde a la pregunta utilizando únicamente la información de los siguientes "
           "pasajes. Si los pasajes no contienen la respuesta, indícalo explícitamente. No "
           "añadas información que no aparezca en los pasajes."),
    "en": ("Answer the question using only the information in the following passages. If the "
           "passages do not contain the answer, say so explicitly. Do not add information that "
           "does not appear in the passages."),
}
_HEADERS = {"es": ("Pasajes", "Pregunta"), "en": ("Passages", "Question")}


def memerag_user_content(passages, query: str, lang: str = "es") -> str:
    """Mensaje de usuario: instrucción + pasajes numerados + pregunta, en el idioma del
    benchmark. Es la P de la partición de TOHA (el contexto al que la respuesta debería
    atender)."""
    p_hdr, q_hdr = _HEADERS[lang]
    blocks = "\n".join(f"{i + 1}: {p['text'].strip()}" for i, p in enumerate(passages))
    return (
        f"{MEMERAG_INSTRUCTION[lang]}\n\n"
        f"{p_hdr}:\n{blocks}\n\n"
        f"{q_hdr}: {query.strip()}"
    )


def _sentence_factuality(fact) -> str:
    """`factuality` es un string (variantes `memerag` y de voto de mayoría) o una lista
    por anotador (variante `_ext`); en ese caso se toma el voto de mayoría."""
    if isinstance(fact, str):
        return fact
    from collections import Counter
    return Counter(fact).most_common(1)[0][0]


def _response_label(answer) -> int:
    """1 si alguna frase es «Not Supported»; 0 en otro caso («Challenging to determine»
    cuenta como 0: no es una afirmación falsa, análogo a la abstención del juez 3-cat)."""
    cats = [_sentence_factuality(s["factuality"]) for s in answer]
    return 1 if any(c == "Not Supported" for c in cats) else 0


def _path(lang: str, variant: str):
    return config.BENCHMARKS_DIR / "memerag-repo" / "data" / variant / f"{lang}.jsonl"


def load_memerag(lang: str = "es", variant: str = "memerag", top_k: int = 4,
                 max_samples: int | None = None) -> list[Sample]:
    """Muestras de MEMERAG en el idioma `lang` (es/en/de/fr/hi; aquí se usan es y en
    para la comparación cross-language RAG). `variant` ∈ {memerag (250, 1 anotador),
    memerag_ext_w_majority_vote (150, voto de mayoría)}. `top_k` recorta los pasajes
    (4 por defecto, para encajar el contexto en memoria sin OOM en el M4)."""
    rows = [json.loads(l) for l in open(_path(lang, variant), encoding="utf-8")]
    if max_samples is not None:
        rows = rows[:max_samples]
    samples = []
    for r in rows:
        passages = r["context"][:top_k]
        answer = " ".join(s["sentence"].strip() for s in r["answer"])
        label = _response_label(r["answer"])
        samples.append(
            Sample(
                id=f"memerag-{lang}-{r['query_id']}",
                prompt=memerag_user_content(passages, r["query"], lang),
                respuesta=answer,
                label=label,
                lang=lang,
                source="rag-notsupported" if label == 1 else "rag-supported",
                meta={"chat": "plain", "variant": variant, "top_k": top_k},
            )
        )
    return samples
