"""Interfaz común a todos los benchmarks: la dataclass `Sample`.

Cada loader (halueval, much, mushroom) traduce su formato propio a una lista de
`Sample`, de modo que el resto del pipeline (extracción, métodos, evaluación) es
agnóstico del benchmark de origen. Contrato fijado en el diseño §4.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Sample:
    """Una muestra etiquetada.

    - `prompt`: contexto al que la respuesta debería atender (en HaluEval-QA,
      knowledge + question). Es la parte P de la partición de TOHA.
    - `respuesta`: texto cuya factualidad se evalúa. Es la parte R.
    - `label`: 0 factual, 1 alucinada.
    - `source`: etiqueta de procedencia dentro del benchmark (p. ej. "right" /
      "hallucinated"), útil para depurar y para análisis por subgrupo.
    """

    id: str
    prompt: str
    respuesta: str
    label: int
    lang: str
    source: str
    meta: dict[str, Any] = field(default_factory=dict)
