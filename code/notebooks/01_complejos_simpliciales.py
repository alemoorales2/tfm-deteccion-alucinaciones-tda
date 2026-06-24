"""
Generación de figuras para la Sección 2.2 del TFM:
- Nube de puntos muestreada de un círculo
- Complejo de Čech a distintos radios
- Complejo de Vietoris-Rips a distintos radios

Requiere: pip install numpy matplotlib scipy ripser gudhi
"""

import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.spatial.distance import pdist, squareform

# Fijar semilla para reproducibilidad
np.random.seed(42)

# --- 1. Generar nube de puntos sobre un círculo con ruido ---
n_points = 30
theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
noise = np.random.normal(0, 0.08, size=(n_points, 2))
points = np.column_stack([np.cos(theta), np.sin(theta)]) + noise

# Matriz de distancias
dist_matrix = squareform(pdist(points))


def get_vr_simplices(points, dist_matrix, epsilon):
    """Calcula los símplices del complejo de Vietoris-Rips para un epsilon dado."""
    n = len(points)
    edges = []
    triangles = []

    # 1-símplices (aristas): distancia < 2*epsilon
    for i, j in combinations(range(n), 2):
        if dist_matrix[i, j] < 2 * epsilon:
            edges.append((i, j))

    # 2-símplices (triángulos): todas las aristas del triángulo presentes
    edge_set = set(edges)
    for i, j, k in combinations(range(n), 3):
        if (min(i,j), max(i,j)) in edge_set and \
           (min(i,k), max(i,k)) in edge_set and \
           (min(j,k), max(j,k)) in edge_set:
            triangles.append((i, j, k))

    return edges, triangles


def plot_complex(ax, points, epsilon, title):
    """Dibuja la nube de puntos con el complejo de Vietoris-Rips superpuesto."""
    edges, triangles = get_vr_simplices(points, dist_matrix, epsilon)

    # Dibujar 2-símplices (triángulos rellenos)
    for tri in triangles:
        triangle = plt.Polygon(points[list(tri)], alpha=0.15, color='steelblue')
        ax.add_patch(triangle)

    # Dibujar 1-símplices (aristas)
    for i, j in edges:
        ax.plot([points[i, 0], points[j, 0]],
                [points[i, 1], points[j, 1]],
                'steelblue', linewidth=0.7, alpha=0.6)

    # Dibujar 0-símplices (puntos)
    ax.scatter(points[:, 0], points[:, 1], c='black', s=20, zorder=5)

    ax.set_title(title, fontsize=13)
    ax.set_aspect('equal')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')


# --- 2. Generar figura con 4 paneles: nube + 3 valores de epsilon ---
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Panel 1: Solo la nube de puntos
axes[0].scatter(points[:, 0], points[:, 1], c='black', s=20)
axes[0].set_title('Nube de puntos $S$', fontsize=13)
axes[0].set_aspect('equal')
axes[0].set_xlim(-1.5, 1.5)
axes[0].set_ylim(-1.5, 1.5)
axes[0].axis('off')

# Panel 2: epsilon pequeño (pocas conexiones)
plot_complex(axes[1], points, epsilon=0.18, title=r'VR$(S,\, \varepsilon=0.18)$')

# Panel 3: epsilon medio (se ve el ciclo)
plot_complex(axes[2], points, epsilon=0.28, title=r'VR$(S,\, \varepsilon=0.28)$')

# Panel 4: epsilon grande (todo relleno)
plot_complex(axes[3], points, epsilon=0.45, title=r'VR$(S,\, \varepsilon=0.45)$')

plt.tight_layout()
plt.savefig('../../memoria/figuras/cap2/vietoris_rips_circulo.png', dpi=200, bbox_inches='tight')
plt.savefig('../../memoria/figuras/cap2/vietoris_rips_circulo.pdf', bbox_inches='tight')
print("Figuras guardadas en memoria/figuras/cap2/")
plt.show()


# --- 3. Figura adicional: solo nube + bolas + complejo de Čech (3 paneles) ---
fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4.2))

eps_cech = 0.28

# Panel 1: Nube de puntos
axes2[0].scatter(points[:, 0], points[:, 1], c='black', s=20)
axes2[0].set_title('Nube de puntos $S$', fontsize=13)
axes2[0].set_aspect('equal')
axes2[0].set_xlim(-1.5, 1.5)
axes2[0].set_ylim(-1.5, 1.5)
axes2[0].axis('off')

# Panel 2: Nube con bolas de radio epsilon
axes2[1].scatter(points[:, 0], points[:, 1], c='black', s=20, zorder=5)
for p in points:
    circle = plt.Circle(p, eps_cech, alpha=0.08, color='steelblue')
    axes2[1].add_patch(circle)
axes2[1].set_title(rf'Bolas de radio $\varepsilon={eps_cech}$', fontsize=13)
axes2[1].set_aspect('equal')
axes2[1].set_xlim(-1.5, 1.5)
axes2[1].set_ylim(-1.5, 1.5)
axes2[1].axis('off')

# Panel 3: Complejo resultante
plot_complex(axes2[2], points, epsilon=eps_cech,
             title=rf'$\check{{C}}(S,\, \varepsilon={eps_cech})$')

plt.tight_layout()
plt.savefig('../../memoria/figuras/cap2/cech_circulo.png', dpi=200, bbox_inches='tight')
plt.savefig('../../memoria/figuras/cap2/cech_circulo.pdf', bbox_inches='tight')
print("Figuras de Čech guardadas en memoria/figuras/cap2/")
plt.show()
