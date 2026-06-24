"""TOHA: divergencia topológica por cabeza (etapa CON modelo).

Para una muestra, recorre las L x H cabezas y calcula la MTop-Div normalizada
por |R| (Cap. 4 §4.3, d_ij(s) = MTop-Div / |R|). El resultado es el intermedio
compacto que se persiste: un escalar por cabeza, [L, H]. La selección de
hallucination-aware heads y la puntuación final (Cap. 4 §4.5-4.6) son etapas SIN
modelo y se implementan en `methods/toha_score.py` (Fase 1, paso F1.4).
"""

import numpy as np

from ..topology.persistence import mtop_div_h0


def sample_head_divergences(attn: np.ndarray, n_prompt: int) -> np.ndarray:
    """Divergencias normalizadas [L, H] de una muestra.

    `attn` es la atención [L, H, T, T] de la muestra; `n_prompt` la frontera P/R.
    """
    L, H, T, _ = attn.shape
    n_resp = T - n_prompt
    div = np.zeros((L, H), dtype=np.float32)
    for l in range(L):
        for h in range(H):
            div[l, h] = mtop_div_h0(attn[l, h], n_prompt)
    if n_resp > 0:
        div /= n_resp
    return div
