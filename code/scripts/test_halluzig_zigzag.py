"""Test de validación del zigzag y la vectorización de HalluZig (Fase 3.A).

Comprueba, sin modelo, las dos piezas de mayor riesgo:

  1. `zigzag_persistence` sobre los cinco grafos sintéticos del notebook
     05_pipeline_halluzig.py (7 nodos, evolución topológica conocida a mano).
     El barcode esperado se razonó en el spike de diseño:
       - H0: la componente secundaria se resepara en las capas puras (G1,G2,G3)
         y se fusiona en las uniones -> barras finitas (0,1),(2,3),(4,5), más la
         componente principal (0,inf).
       - H1: ciclo 0-1-2 nace en G1∪G2 (paso 1) y muere en G4 (paso 6) -> (1,6);
         ciclo 3-4-5 nace en G2∪G3 (paso 3) y persiste -> (3,inf); ciclo grande
         que cierra la arista 6-0 de G5 -> (7,inf).
  2. Formas y propiedades de la vectorización (PI 20x20, entropía, Betti 50).

    .venv/bin/python code/scripts/test_halluzig_zigzag.py
"""

import sys

import numpy as np

from tfm_hallu.topology import (
    betti_curve,
    build_layer_graphs,
    persistence_entropy,
    persistence_image,
    zigzag_persistence,
)

# Los cinco grafos sintéticos (T=7) del notebook 05.
SYNTH = [
    [(0, 1), (1, 2), (3, 4), (4, 5), (5, 6)],
    [(0, 1), (1, 2), (2, 0), (2, 3), (4, 5), (5, 6)],
    [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)],
    [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 3), (5, 6)],
    [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 3), (5, 6), (6, 0)],
]


def _barset(bars):
    """Barras como conjunto de tuplas (birth, 'inf' | death), comparable sin orden."""
    return {
        (round(float(b), 6), "inf" if not np.isfinite(dth) else round(float(dth), 6))
        for b, dth in bars
    }


def test_zigzag_synthetic():
    layers = [set(tuple(sorted(e)) for e in g) for g in SYNTH]
    barcode, n_levels = zigzag_persistence(layers, n_nodes=7)
    assert n_levels == 2 * 5 - 1 == 9, n_levels

    got_h0, got_h1 = _barset(barcode[0]), _barset(barcode[1])
    exp_h0 = {(0.0, 1.0), (2.0, 3.0), (4.0, 5.0), (0.0, "inf")}
    exp_h1 = {(1.0, 6.0), (3.0, "inf"), (7.0, "inf")}
    assert got_h0 == exp_h0, f"H0 inesperado: {got_h0}"
    assert got_h1 == exp_h1, f"H1 inesperado: {got_h1}"
    print(f"  [OK] zigzag sintético: H0={got_h0}")
    print(f"                         H1={got_h1}")


def test_build_layer_graphs():
    rng = np.random.default_rng(0)
    L, H, T = 4, 3, 10
    attn = rng.random((L, H, T, T))
    graphs, t = build_layer_graphs(attn, percentile=90.0)
    assert t == T and len(graphs) == L
    n_pairs = T * (T - 1) // 2
    # percentil 90 -> ~10% de los pares; tolerancia amplia por ruido
    for g in graphs:
        assert 0 < len(g) <= max(1, int(0.25 * n_pairs)), len(g)
        assert all(i < j for (i, j) in g)
    print(f"  [OK] build_layer_graphs: {[len(g) for g in graphs]} aristas/capa (de {n_pairs} pares)")


def test_vectorization():
    layers = [set(tuple(sorted(e)) for e in g) for g in SYNTH]
    barcode, n_levels = zigzag_persistence(layers, n_nodes=7)
    max_level = n_levels - 1

    pi = persistence_image(barcode[1], max_level, resolution=20, sigma=0.1)
    ent = persistence_entropy(barcode[1], max_level)
    bet = betti_curve(barcode[1], max_level, n_points=50)
    assert pi.shape == (400,) and np.all(pi >= 0) and pi.sum() > 0, pi.shape
    assert np.isscalar(ent) or ent.ndim == 0
    assert bet.shape == (50,) and bet.max() >= 1
    # vector completo HalluZig: (PI 400 + entropía 1 + Betti 50) x (H0, H1) = 902
    vec = np.concatenate([
        persistence_image(barcode[d], max_level, 20, 0.1) for d in (0, 1)
    ] + [
        np.array([persistence_entropy(barcode[d], max_level)], np.float32) for d in (0, 1)
    ] + [
        betti_curve(barcode[d], max_level, 50) for d in (0, 1)
    ])
    assert vec.shape == (902,), vec.shape

    # barcode vacío -> todo ceros (sin errores)
    empty = np.zeros((0, 2))
    assert persistence_image(empty, max_level).sum() == 0
    assert float(persistence_entropy(empty, max_level)) == 0.0
    assert betti_curve(empty, max_level).sum() == 0
    print(f"  [OK] vectorización: PI sum(H1)={pi.sum():.3f}, entropía(H1)={float(ent):.3f}, "
          f"Betti(H1) máx={int(bet.max())}; vector D={vec.shape[0]}")


def main():
    print("=== Test HalluZig: zigzag + vectorización ===")
    test_zigzag_synthetic()
    test_build_layer_graphs()
    test_vectorization()
    print("\nTodas las comprobaciones OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
