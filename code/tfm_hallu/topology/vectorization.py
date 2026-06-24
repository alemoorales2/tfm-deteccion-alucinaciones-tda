"""Vectorización de barcodes (Cap. 4 §4.3.4, Cap. 5 §5.1.4).

Convierte un barcode (lista de pares birth/death) en descriptores de dimensión
fija para alimentar un clasificador: imagen de persistencia, entropía de
persistencia y curva de Betti. HalluZig concatena los tres en H_0 y H_1.

El eje de la filtración zigzag (pasos 0..2L-2) se normaliza a [0,1] dividiendo
por el último paso, de modo que el vector resultante es **independiente del
modelo** (mismo tamaño y rango aunque L cambie entre arquitecturas), lo que es
clave para la transferibilidad cross-model (Fase 5). Las barras infinitas se
recortan al último paso antes de normalizar.
"""

import numpy as np
from persim import PersistenceImager


def normalize_bars(bars: np.ndarray, max_level: float) -> np.ndarray:
    """Recorta muertes infinitas a `max_level` y normaliza (birth, death) a [0,1]."""
    if bars.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    b = bars.astype(np.float64).copy()
    b[~np.isfinite(b[:, 1]), 1] = max_level
    div = max_level if max_level > 0 else 1.0
    return b / div


def persistence_image(bars: np.ndarray, max_level: float, resolution: int = 20,
                      sigma: float = 0.1) -> np.ndarray:
    """Imagen de persistencia `resolution`×`resolution` (aplanada) sobre el
    cuadrado [0,1]². `sigma` (desviación del kernel gaussiano, en unidades del
    eje normalizado) se fija constante: no se calibra por fold, para preservar la
    independencia del intermedio por muestra y evitar fuga."""
    n = resolution * resolution
    b = normalize_bars(bars, max_level)
    if b.shape[0] == 0:
        return np.zeros(n, dtype=np.float32)
    pimgr = PersistenceImager(
        birth_range=(0.0, 1.0),
        pers_range=(0.0, 1.0),
        pixel_size=1.0 / resolution,
        kernel_params={"sigma": [[sigma ** 2, 0.0], [0.0, sigma ** 2]]},
    )
    img = np.asarray(pimgr.transform(b))           # recibe (birth, death)
    return img.ravel().astype(np.float32)


def persistence_entropy(bars: np.ndarray, max_level: float) -> float:
    """Entropía de Shannon de las longitudes de vida normalizadas (Cap. 5 §5.5.2)."""
    b = normalize_bars(bars, max_level)
    if b.shape[0] == 0:
        return np.float32(0.0)
    life = b[:, 1] - b[:, 0]
    life = life[life > 0]
    if life.size == 0:
        return np.float32(0.0)
    p = life / life.sum()
    return np.float32(-(p * np.log(p)).sum())


def betti_curve(bars: np.ndarray, max_level: float, n_points: int = 50) -> np.ndarray:
    """Curva de Betti muestreada en `n_points` puntos equiespaciados de [0,1]:
    cuántas barras están vivas en cada punto de la filtración."""
    b = normalize_bars(bars, max_level)
    xs = np.linspace(0.0, 1.0, n_points)
    curve = np.zeros(n_points, dtype=np.float32)
    for birth, death in b:
        curve += ((xs >= birth) & (xs < death)).astype(np.float32)
    return curve
