"""Métodos de detección. Cada uno deriva su intermedio compacto por muestra."""

from .halluzig import sample_halluzig_features
from .lapeigvals import sample_lap_eigenvalues
from .toha import sample_head_divergences

__all__ = [
    "sample_head_divergences",
    "sample_lap_eigenvalues",
    "sample_halluzig_features",
]
