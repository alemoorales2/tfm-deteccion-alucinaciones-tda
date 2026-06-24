"""Loader de HaluEval-QA.

Formato de origen (un objeto JSON por línea, 10 000 líneas):
    {knowledge, question, right_answer, hallucinated_answer}

Cada ítem produce DOS muestras etiquetadas que comparten el mismo prompt
(knowledge + question): la respuesta correcta (label 0) y la alucinada
(label 1). El conjunto queda balanceado por construcción.

Adaptación documentada (diseño §10): las respuestas vienen dadas por el
benchmark, no las genera el modelo evaluado. TOHA se aplica aquí en su versión
estática (Cap. 4 §4.3): un único forward pass sobre [prompt + respuesta] y la
divergencia topológica entre R (respuesta) y P (prompt).
"""

import json
import random
from pathlib import Path

from .. import config
from .base import Sample


def _default_path() -> Path:
    return config.BENCHMARKS_DIR / "halueval-qa" / "qa_data.json"


def _format_prompt(knowledge: str, question: str) -> str:
    """Contexto + pregunta como un único prompt (parte P). Concatenación cruda,
    sin plantilla de chat: model-agnóstica para el barrido de cuatro modelos."""
    return f"{knowledge.strip()}\n\n{question.strip()}"


def load_halueval_qa(
    max_samples: int | None = 1000,
    seed: int = config.SEED,
    path: Path | None = None,
) -> list[Sample]:
    """Carga HaluEval-QA como lista de `Sample`.

    `max_samples` es el número de MUESTRAS objetivo (no de ítems); como cada
    ítem da 2 muestras, se muestrean `max_samples // 2` ítems con `seed`. None
    usa el benchmark entero (20 000 muestras).
    """
    path = path or _default_path()
    with path.open(encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    if max_samples is not None:
        n_items = min(max_samples // 2, len(items))
        items = random.Random(seed).sample(items, n_items)

    samples: list[Sample] = []
    for i, it in enumerate(items):
        prompt = _format_prompt(it["knowledge"], it["question"])
        base_id = f"halueval-qa-{i:05d}"
        samples.append(
            Sample(
                id=f"{base_id}-r",
                prompt=prompt,
                respuesta=it["right_answer"].strip(),
                label=0,
                lang="en",
                source="right",
                meta={"item": i},
            )
        )
        samples.append(
            Sample(
                id=f"{base_id}-h",
                prompt=prompt,
                respuesta=it["hallucinated_answer"].strip(),
                label=1,
                lang="en",
                source="hallucinated",
                meta={"item": i},
            )
        )
    return samples
