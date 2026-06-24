"""Homología persistente para TOHA: MTop-Div de dimensión 0.

Implementa la divergencia topológica del Cap. 4 §4.3, que es la Manifold Topology
Divergence (cross-barcode de Barannikov et al., 2021) adaptada por TOHA al par
prompt (P) / respuesta (R):

  - Transformación atención -> distancia: d_ij = 1 - max(A_ij, A_ji), en [0,1]
    (Cap. 4, Ec. atencion_distancia, simetrizada por el máximo).
  - Cross-barcode con P como referencia: las distancias DENTRO de P se ponen a 0
    (P se colapsa a un único bloque); las distancias R-P y R-R conservan 1 - A.
  - Se calcula la homología persistente de H0. Las |R| barras finitas miden el
    coste de adherir cada token de la respuesta al bloque P (directamente o a
    través de otro token de R). MTop-Div = suma de sus longitudes (b_i - a_i);
    la clase esencial del bloque P no suma.

Esta es la lectura correcta de «el bosque generador mínimo que conecta R con P»
(Cap. 4): contraído P a un nodo, MTop-Div es el peso del árbol generador mínimo
sobre {P} ∪ R. Crece como O(|R|), de modo que d_ij = MTop-Div / |R| es un coste
medio de adhesión acotado, robusto a la longitud de la respuesta.
"""

import numpy as np
from ripser import ripser


def mtop_div_h0(attn_2d: np.ndarray, n_prompt: int) -> float:
    """MTop-Div de H0 de una cabeza (cross-barcode P-colapsado).

    `attn_2d` es la matriz de atención [T, T] de una cabeza; `n_prompt` el número
    de tokens de P (los primeros del eje). Devuelve la divergencia SIN normalizar
    (la normalización por |R| la aplica el método TOHA).
    """
    T = attn_2d.shape[0]
    n_resp = T - n_prompt
    if n_prompt <= 0 or n_resp <= 0:
        return 0.0

    a = np.asarray(attn_2d, dtype=np.float64)
    a = np.maximum(a, a.T)  # simetrización por el máximo
    d = 1.0 - a

    # Cross-barcode: P (referencia) se colapsa a distancia 0; R-P y R-R intactas.
    d[:n_prompt, :n_prompt] = 0.0
    np.fill_diagonal(d, 0.0)

    dgm0 = ripser(d, maxdim=0, distance_matrix=True)["dgms"][0]
    finite = dgm0[np.isfinite(dgm0[:, 1])]
    return float(np.sum(finite[:, 1] - finite[:, 0]))
