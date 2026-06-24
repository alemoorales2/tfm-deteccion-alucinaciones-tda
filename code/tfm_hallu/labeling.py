"""Etiquetado híbrido de respuestas generadas: exact-match + juez Gemini.

- exact-match (normalizado, por palabras completas) contra la respuesta de
  referencia: si aparece, la respuesta es CORRECTA (label 0). Es el caso claro.
- el resto (la referencia no aparece literalmente) es ambiguo (puede ser
  paráfrasis correcta o alucinación) y lo decide el juez Gemini.

El juez se llama en lotes (batching) con salida JSON, para respetar el límite de
peticiones del plan gratuito. Los lotes se espacian en el tiempo (throttle).
"""

import json
import re
import time
import unicodedata

import pandas as pd

from . import config

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_NONALNUM = re.compile(r"[^a-z0-9 ]")


def normalize(s: str) -> list[str]:
    s = s.lower()
    s = _ARTICLES.sub(" ", s)
    s = _NONALNUM.sub(" ", s)
    return s.split()


_ARTICLES_ES = re.compile(r"\b(el|la|los|las|un|una|unos|unas|lo)\b")


def _strip_accents(s: str) -> str:
    """Quita diacríticos (á→a, ñ→n, ü→u) descomponiendo en NFD y descartando las
    marcas combinantes. Se aplica por igual a la respuesta dorada y a la candidata,
    de modo que un acento de más o de menos no rompe el emparejamiento."""
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def normalize_es(s: str) -> list[str]:
    """Variante española de `normalize`. La `normalize` anglocéntrica borra los
    acentos y la ñ como caracteres no alfanuméricos (corrompiendo el español); esta
    los normaliza antes de filtrar y elimina los artículos españoles en vez de los
    ingleses, para un exact-match fiel en ES."""
    s = _strip_accents(s.lower())
    s = _ARTICLES_ES.sub(" ", s)
    s = _NONALNUM.sub(" ", s)
    return s.split()


def exact_match(gold: str, gen: str) -> bool:
    """True si los tokens normalizados de `gold` aparecen como subsecuencia
    contigua de palabras en `gen` (evita falsos positivos de subcadena)."""
    g = normalize(gold)
    t = normalize(gen)
    if not g:
        return False
    return any(t[i : i + len(g)] == g for i in range(len(t) - len(g) + 1))


_JUDGE_INSTR = (
    "Eres un evaluador de un sistema de pregunta-respuesta. Para cada ítem se da una "
    "pregunta, la respuesta de referencia correcta (gold) y una respuesta candidata "
    "generada por un modelo. Clasifica la candidata en una de tres categorías:\n"
    "- CORRECTA: coincide en significado con la referencia. Acepta paráfrasis, "
    "abreviaciones y variantes equivalentes (p. ej. «Ron Shelton» = «Ronald Wayne Ron "
    "Shelton»; «Douglas SBD» = «Douglas SBD Dauntless»).\n"
    "- ABSTENCION: la candidata NO afirma una respuesta; reconoce no saber o no poder "
    "verificar (p. ej. «no lo sé», «no puedo verificar», «no encontré información»). No "
    "inventa nada.\n"
    "- ALUCINACION: afirma una respuesta concreta que NO coincide con la referencia "
    "(información incorrecta, inventada o no respaldada).\n"
    "Una abstención NO es una alucinación. Devuelve un array JSON, un objeto por ítem: "
    '{"id": <int>, "verdict": "CORRECTA"|"ABSTENCION"|"ALUCINACION"}.'
)


def _gemini_client():
    """Cliente Gemini, solo si el juez es Gemini; en otro backend devuelve None (no se
    exige GEMINI_KEY). Los call-sites pasan este valor a `judge_batch`, que lo ignora
    cuando el backend no es Gemini."""
    if config.JUDGE_BACKEND != "gemini":
        return None
    key = config.load_env_var("GEMINI_KEY")
    if not key:
        raise RuntimeError("GEMINI_KEY no encontrada en .env")
    from google import genai

    return genai.Client(api_key=key)


# Esquema de salida para el juez local (Ollama): fuerza el array de veredictos. Sin él,
# los modelos pequeños devuelven JSON degenerado o incompleto (verificado en el test).
_OLLAMA_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "verdict": {"type": "string", "enum": ["CORRECTA", "ABSTENCION", "ALUCINACION"]},
        },
        "required": ["id", "verdict"],
    },
}


def _judge_batch_gemini(client, items, model=None) -> dict:
    from google.genai import types

    model = model or config.GEMINI_JUDGE_MODEL
    body = "\n".join(f"[{i}] Pregunta: {q} | Referencia: {r} | Candidata: {c}" for i, q, r, c in items)
    resp = client.models.generate_content(
        model=model,
        contents=f"{_JUDGE_INSTR}\n\n{body}",
        config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json"),
    )
    data = json.loads(resp.text)
    return {int(d["id"]): str(d["verdict"]).strip().upper().startswith("ALUC") for d in data}


def _judge_batch_ollama(items, model=None) -> dict:
    """Juez local vía Ollama (/api/generate) con salida forzada por esquema JSON."""
    import urllib.request

    model = model or config.OLLAMA_JUDGE_MODEL
    body = "\n".join(f"[{i}] Pregunta: {q} | Referencia: {r} | Candidata: {c}" for i, q, r, c in items)
    payload = {
        "model": model,
        "prompt": f"{_JUDGE_INSTR}\n\n{body}",
        "stream": False,
        "format": _OLLAMA_SCHEMA,
        "options": {"temperature": 0},
    }
    req = urllib.request.Request(
        config.OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    resp = json.loads(urllib.request.urlopen(req, timeout=config.OLLAMA_TIMEOUT).read())
    data = json.loads(resp["response"])
    return {int(d["id"]): str(d["verdict"]).strip().upper().startswith("ALUC") for d in data}


def judge_batch(client, items, model=None) -> dict:
    """Juzga un lote. `items`: lista de (id, pregunta, referencia, candidata).
    Devuelve {id: is_hallucination(bool)}. Despacha al backend de `config.JUDGE_BACKEND`
    (juez local Ollama por defecto; Gemini si se configura). `client` solo lo usa el
    backend Gemini; con Ollama se ignora (y el `model` de Gemini también)."""
    if config.JUDGE_BACKEND == "ollama":
        return _judge_batch_ollama(items)
    return _judge_batch_gemini(client, items, model=model)


def _load_verdicts(path) -> dict:
    if path is not None and path.exists():
        vc = pd.read_parquet(path)
        return {int(k): bool(v) for k, v in zip(vc["item"], vc["is_hall"])}
    return {}


def _save_verdicts(path, cache: dict):
    if path is not None:
        pd.DataFrame({"item": list(cache), "is_hall": list(cache.values())}).to_parquet(path, index=False)


def hybrid_label(records, verdict_cache_path=None, batch_size=None, sleep=None, model=None):
    """Etiquetado híbrido REANUDABLE de dicts con claves item/question/right_answer/generated.

    exact-match → factual (0); el resto lo decide el juez Gemini en lotes. Los
    veredictos se cachean por `item` en `verdict_cache_path` y se guardan tras cada
    lote, de modo que un corte por cuota (429) no pierde lo ya juzgado: al re-ejecutar
    se continúa solo con lo pendiente.

    Devuelve (labels, stats). labels[i] ∈ {0,1} o None si quedó sin juzgar (cuota).
    stats: {ambiguous, judged_now, remaining, quota_hit}.
    """
    batch_size = batch_size or config.JUDGE_BATCH_SIZE
    sleep = config.JUDGE_SLEEP_SECONDS if sleep is None else sleep

    cache = _load_verdicts(verdict_cache_path)
    labels = [None] * len(records)
    ambiguous = []
    for i, r in enumerate(records):
        if exact_match(r["right_answer"], r["generated"]):
            labels[i] = 0
        else:
            ambiguous.append(i)

    pending = []
    for i in ambiguous:
        it = int(records[i]["item"])
        if it in cache:
            labels[i] = 1 if cache[it] else 0
        else:
            pending.append(i)

    judged_now, quota_hit = 0, False
    if pending:
        client = _gemini_client()
        for b in range(0, len(pending), batch_size):
            chunk = pending[b : b + batch_size]
            items = [
                (int(records[i]["item"]), records[i]["question"], records[i]["right_answer"], records[i]["generated"])
                for i in chunk
            ]
            try:
                verdicts = judge_batch(client, items, model=model)
            except Exception as e:  # noqa: BLE001  (cuota u otro error: parar y guardar)
                print(f"  [STOP] juez interrumpido ({type(e).__name__}): {str(e)[:140]}")
                quota_hit = True
                break
            for i in chunk:
                it = int(records[i]["item"])
                ish = bool(verdicts.get(it, True))  # ante duda, alucinación
                cache[it] = ish
                labels[i] = 1 if ish else 0
            judged_now += len(chunk)
            _save_verdicts(verdict_cache_path, cache)
            if b + batch_size < len(pending):
                time.sleep(sleep)

    remaining = sum(1 for v in labels if v is None)
    return labels, {"ambiguous": len(ambiguous), "judged_now": judged_now, "remaining": remaining, "quota_hit": quota_hit}
