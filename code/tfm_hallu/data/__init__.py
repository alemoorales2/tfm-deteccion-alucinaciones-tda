"""Loaders de benchmarks. Cada uno devuelve una lista de `Sample`."""

from .base import Sample
from .halueval import load_halueval_qa
from .halueval_gen import load_halueval_qa_generated
from .memerag import load_memerag
from .much import load_much
from .mushroom import load_mushroom_es
from .squad import load_squad
from .xquad import load_xquad

__all__ = [
    "Sample",
    "load_halueval_qa",
    "load_halueval_qa_generated",
    "load_memerag",
    "load_much",
    "load_mushroom_es",
    "load_squad",
    "load_xquad",
]
