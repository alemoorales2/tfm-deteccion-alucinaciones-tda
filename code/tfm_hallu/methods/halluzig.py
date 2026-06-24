"""HalluZig: firma topológica de la evolución de la atención (etapa CON modelo).

Reimplementa HalluZig (Samaga et al. 2026) según el Cap. 4 §4.3 y el Cap. 5
§5.1.4. De cada muestra deriva un vector de dimensión fija:

  1. Grafos de atención por capa: media de cabezas + umbral por percentil
     (`topology.build_layer_graphs`).
  2. Persistencia zigzag de la sucesión de grafos en H0 y H1
     (`topology.zigzag_persistence`, backend Dionysus 2).
  3. Vectorización por dimensión homológica: imagen de persistencia 20x20 +
     entropía + curva de Betti (50 pts), con el eje de la filtración normalizado
     a [0,1] (`topology.vectorization`).

El vector resultante (D=902) es independiente del modelo: mismo tamaño y rango
aunque cambie L entre arquitecturas, lo que habilita la transferencia cross-model
de la Fase 5 sin alinear dimensiones.
"""

import numpy as np

from .. import config
from ..topology import (
    betti_curve,
    build_layer_graphs,
    persistence_entropy,
    persistence_image,
    zigzag_persistence,
)


def halluzig_feature_dim(pi_res: int | None = None, betti_points: int | None = None) -> int:
    """Dimensión del vector de features (PI + entropía + Betti) x (H0, H1)."""
    pi_res = config.HALLUZIG_PI_RES if pi_res is None else pi_res
    betti_points = config.HALLUZIG_BETTI_POINTS if betti_points is None else betti_points
    return 2 * (pi_res * pi_res + 1 + betti_points)


def sample_halluzig_features(
    attn: np.ndarray,
    percentile: float | None = None,
    pi_res: int | None = None,
    pi_sigma: float | None = None,
    betti_points: int | None = None,
    return_barcode: bool = False,
):
    """Vector de features HalluZig de una muestra a partir de su atención [L,H,T,T].

    Con `return_barcode=True` devuelve `(features, barcode, n_levels)` donde
    `barcode = {0: [m0,2], 1: [m1,2]}` son los pares (birth, death) en el eje de
    pasos del zigzag (sin normalizar; death puede ser inf). Cachear el barcode
    permite re-vectorizar (filtro de barras cortas, resolución de la imagen, etc.)
    sin re-extraer del modelo, que es lo caro a contexto largo (RAG)."""
    percentile = config.HALLUZIG_PERCENTILE if percentile is None else percentile
    pi_res = config.HALLUZIG_PI_RES if pi_res is None else pi_res
    pi_sigma = config.HALLUZIG_PI_SIGMA if pi_sigma is None else pi_sigma
    betti_points = config.HALLUZIG_BETTI_POINTS if betti_points is None else betti_points

    graphs, n_nodes = build_layer_graphs(attn, percentile)
    barcode, n_levels = zigzag_persistence(graphs, n_nodes)
    max_level = n_levels - 1

    parts = []
    for dim in (0, 1):
        bars = barcode[dim]
        parts.append(persistence_image(bars, max_level, pi_res, pi_sigma))
        parts.append(np.array([persistence_entropy(bars, max_level)], dtype=np.float32))
        parts.append(betti_curve(bars, max_level, betti_points))
    feats = np.concatenate(parts).astype(np.float32)
    if return_barcode:
        return feats, barcode, n_levels
    return feats
