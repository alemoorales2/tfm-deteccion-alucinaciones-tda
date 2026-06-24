"""LapEigvals: autovalores del Laplaciano de la atención (etapa CON modelo).

Reimplementa el método de Binkowski et al. (2025) tal como lo describe el Cap. 5
§5.4. Cada matriz de atención A^{(l,h)} se interpreta como matriz de adyacencia de
un grafo dirigido. El Laplaciano dirigido con grado de salida normalizado es

    L^{(l,h)} = D^{(l,h)} - A^{(l,h)},   d_ii = (sum_u a_ui) / (T - i),

donde la suma recorre la columna i (atención que el token i recibe de los que le
siguen) y T - i es el número de tokens posteriores. Como la atención causal es
triangular inferior, L también lo es y sus autovalores son su diagonal
(Eq. 3 del paper): no se diagonaliza nada. Por cabeza se toman los k autovalores
mayores; la concatenación sobre capas y cabezas da un vector L*H*k.
"""

import numpy as np


def sample_lap_eigenvalues(attn: np.ndarray, k: int) -> np.ndarray:
    """Autovalores [L, H, k] de una muestra (k mayores por cabeza).

    `attn` es la atención [L, H, T, T], triangular inferior (causal).
    """
    L, H, T, _ = attn.shape
    out = np.zeros((L, H, k), dtype=np.float32)
    denom = (T - np.arange(T)).astype(np.float64)  # T - i, i = 0..T-1
    for l in range(L):
        for h in range(H):
            a = attn[l, h].astype(np.float64)
            d = a.sum(axis=0) / denom          # grado de salida normalizado
            diag_lap = d - np.diag(a)          # diagonal de L = D - A
            order = np.sort(diag_lap)[::-1]    # autovalores en orden decreciente
            m = min(k, T)
            out[l, h, :m] = order[:m]          # k mayores (pad con 0 si T < k)
    return out
