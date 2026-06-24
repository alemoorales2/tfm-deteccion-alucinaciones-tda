"""Persistencia zigzag para HalluZig (Cap. 4 §4.3).

HalluZig modela la evolución de la topología de la atención a lo largo de las
capas. De cada muestra se construye una sucesión de grafos por capa
$G_1, \\dots, G_L$ (promedio de cabezas + umbral por percentil) y se calcula la
persistencia zigzag de la filtración

    G_1 ⊂ G_1∪G_2 ⊃ G_2 ⊂ G_2∪G_3 ⊃ ... ⊃ G_L,

intercalando los grafos unión entre capas consecutivas. El barcode resultante en
$H_0$ (componentes) y $H_1$ (ciclos) resume esa dinámica.

Backend: Dionysus 2 (verificado en 00_check_env.py). La elección final frente a
FastZigzag se decide por benchmark de tiempos en esta fase (diseño §10): aquí se
implementa con Dionysus, que desbloquea el método sin compilar nada.

Eje de la filtración: el índice de paso del zigzag, 0..2L-2 (2L-1 pasos). El paso
par 2k es el grafo G_{k+1}; el impar 2k+1 es la unión G_{k+1}∪G_{k+2}.
"""

import dionysus as d
import numpy as np


def build_layer_graphs(attn: np.ndarray, percentile: float = 90.0):
    """Grafos de atención por capa para HalluZig.

    `attn` es la atención [L, H, T, T]. Por capa se promedian las cabezas,
    A^{(l)} = (1/H) Σ_h A^{(l,h)}, se simetriza por el máximo (grafo no dirigido;
    como la atención es causal, para i<j el peso es A^{(l)}_{ji}, la atención que
    el token posterior j dedica al anterior i) y se retienen las aristas (i,j),
    i<j, cuyo peso supera el `percentile` de los pesos fuera de la diagonal de la
    capa. Devuelve (lista de L conjuntos de aristas (i,j) con i<j, T).
    """
    L, H, T, _ = attn.shape
    iu = np.triu_indices(T, k=1)  # pares i<j (sin diagonal)
    graphs = []
    for l in range(L):
        A = attn[l].mean(axis=0)        # [T, T]
        A = np.maximum(A, A.T)          # simetrización por el máximo
        w = A[iu]
        if w.size == 0:
            graphs.append(set())
            continue
        thr = np.percentile(w, percentile)
        mask = w > thr
        graphs.append(set(zip(iu[0][mask].tolist(), iu[1][mask].tolist())))
    return graphs, T


def _level_edges(layer_edges, lvl):
    """Aristas presentes en el paso `lvl` del zigzag."""
    if lvl % 2 == 0:                       # paso par 2k -> G_{k+1}
        return layer_edges[lvl // 2]
    k = lvl // 2                           # paso impar 2k+1 -> G_{k+1} ∪ G_{k+2}
    return layer_edges[k] | layer_edges[k + 1]


def _presence_to_times(present: set, n_levels: int):
    """Conjunto de pasos donde un símplice está presente -> lista de tiempos de
    entrada/salida alternados (convención de Dionysus). Longitud impar => sigue
    vivo al final del zigzag (barra infinita)."""
    times, prev = [], False
    for lvl in range(n_levels):
        cur = lvl in present
        if cur and not prev:
            times.append(float(lvl))       # aparece
        if not cur and prev:
            times.append(float(lvl))       # desaparece
        prev = cur
    return times


def zigzag_persistence(layer_edges, n_nodes: int):
    """Persistencia zigzag de la sucesión de grafos `layer_edges`.

    `layer_edges` es la lista de L conjuntos de aristas (i<j) por capa; `n_nodes`
    el número de tokens (vértices, presentes en todos los pasos). Devuelve
    (barcode, n_levels), donde `barcode` es {0: array [m0, 2], 1: array [m1, 2]}
    con pares (birth, death) en el eje de pasos del zigzag; death puede ser inf.
    """
    L = len(layer_edges)
    n_levels = 2 * L - 1
    all_edges = set().union(*layer_edges) if layer_edges else set()
    level_edges = [_level_edges(layer_edges, lvl) for lvl in range(n_levels)]

    simplices, times = [], []
    for v in range(n_nodes):
        simplices.append([v])
        times.append([0.0])                # vértices: presentes desde el inicio
    for (i, j) in all_edges:
        present = {lvl for lvl in range(n_levels) if (i, j) in level_edges[lvl]}
        simplices.append([i, j])
        times.append(_presence_to_times(present, n_levels))

    f = d.Filtration(simplices)
    _, dgms, _ = d.zigzag_homology_persistence(f, times)

    barcode = {}
    for dim in (0, 1):
        bars = []
        if dim < len(dgms):
            bars = [(float(p.birth), float(p.death)) for p in dgms[dim]]
        barcode[dim] = np.array(bars, dtype=np.float64).reshape(-1, 2)
    return barcode, n_levels
