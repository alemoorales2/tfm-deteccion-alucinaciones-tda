"""
Generación de figuras para la Sección 2.3 del TFM:
- Diagrama de persistencia y barcode de dos círculos concéntricos
- Paisaje de persistencia

Datos: dos círculos concéntricos (radios 1 y 2) con ruido gaussiano.
Resultado esperado: H0 detecta 2 componentes que se fusionan,
                    H1 detecta 2 ciclos independientes.

Requiere: pip3 install numpy matplotlib scipy ripser
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    from ripser import ripser
except ImportError:
    print("Instalar con: pip3 install ripser")
    exit(1)

np.random.seed(42)

# --- 1. Generar dos círculos concéntricos con ruido ---
n1, n2 = 40, 50  # puntos por círculo

# Círculo interior (radio 1)
theta1 = np.linspace(0, 2 * np.pi, n1, endpoint=False)
circle1 = np.column_stack([np.cos(theta1), np.sin(theta1)])
circle1 += np.random.normal(0, 0.07, size=circle1.shape)

# Círculo exterior (radio 2.2)
theta2 = np.linspace(0, 2 * np.pi, n2, endpoint=False)
circle2 = 2.2 * np.column_stack([np.cos(theta2), np.sin(theta2)])
circle2 += np.random.normal(0, 0.10, size=circle2.shape)

points = np.vstack([circle1, circle2])

# --- 2. Calcular homología persistente ---
result = ripser(points, maxdim=1)
dgms = result['dgms']

# --- 3. Figura 1: Nube de puntos + Diagramas de persistencia (3 paneles) ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Panel izquierdo: nube de puntos
axes[0].scatter(circle1[:, 0], circle1[:, 1], c='steelblue', s=15, label='Círculo interior')
axes[0].scatter(circle2[:, 0], circle2[:, 1], c='indianred', s=15, label='Círculo exterior')
axes[0].set_title('Nube de puntos $S$', fontsize=13)
axes[0].set_aspect('equal')
axes[0].legend(fontsize=9)
axes[0].axis('off')

# Paneles central y derecho: diagramas H0 y H1
for dim, ax in enumerate(axes[1:]):
    dgm = dgms[dim]
    finite = dgm[dgm[:, 1] < np.inf]
    infinite = dgm[dgm[:, 1] == np.inf]

    max_val = max(finite[:, 1].max() if len(finite) > 0 else 1,
                  finite[:, 0].max() if len(finite) > 0 else 1) * 1.15
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, linewidth=1)

    if len(finite) > 0:
        ax.scatter(finite[:, 0], finite[:, 1], s=40,
                   c='steelblue' if dim == 0 else 'indianred',
                   edgecolors='black', linewidths=0.5, zorder=5)

    if len(infinite) > 0:
        ax.scatter(infinite[:, 0], [max_val * 0.95] * len(infinite), s=80,
                   c='steelblue' if dim == 0 else 'indianred',
                   marker='^', edgecolors='black', linewidths=0.5, zorder=5)

    ax.set_xlabel('Nacimiento', fontsize=11)
    ax.set_ylabel('Muerte', fontsize=11)
    ax.set_title(f'$H_{dim}$', fontsize=13)
    ax.set_xlim(-0.02, max_val)
    ax.set_ylim(-0.02, max_val)
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('../../memoria/figuras/cap2/diagrama_persistencia.png', dpi=200, bbox_inches='tight')
plt.savefig('../../memoria/figuras/cap2/diagrama_persistencia.pdf', bbox_inches='tight')
print("Diagrama de persistencia guardado.")


# --- 4. Figura 2: Barcode ---
fig2, axes2 = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

for dim, ax in enumerate(axes2):
    dgm = dgms[dim]
    order = np.argsort(dgm[:, 0])
    dgm_sorted = dgm[order]

    color = 'steelblue' if dim == 0 else 'indianred'
    max_death = dgm_sorted[dgm_sorted[:, 1] < np.inf][:, 1].max() * 1.1

    for idx, (birth, death) in enumerate(dgm_sorted):
        if death == np.inf:
            ax.plot([birth, max_death], [idx, idx], color=color, linewidth=2, alpha=0.8)
            ax.scatter(max_death, idx, marker='>', color=color, s=25, zorder=5)
        else:
            ax.plot([birth, death], [idx, idx], color=color, linewidth=2, alpha=0.8)

    ax.set_ylabel(f'$H_{dim}$', fontsize=13)
    ax.set_yticks([])

axes2[1].set_xlabel('$\\varepsilon$', fontsize=13)
axes2[0].set_title('Código de barras', fontsize=13)

plt.tight_layout()
plt.savefig('../../memoria/figuras/cap2/barcode.png', dpi=200, bbox_inches='tight')
plt.savefig('../../memoria/figuras/cap2/barcode.pdf', bbox_inches='tight')
print("Barcode guardado.")


# --- 5. Figura 3: Paisaje de persistencia para H1 ---
def persistence_landscape(dgm, k_max=3, num_points=500):
    """Calcula los primeros k_max paisajes de persistencia."""
    finite = dgm[dgm[:, 1] < np.inf]
    if len(finite) == 0:
        return np.zeros((k_max, num_points)), np.linspace(0, 1, num_points)

    births = finite[:, 0]
    deaths = finite[:, 1]

    t_min = births.min()
    t_max = deaths.max()
    t_grid = np.linspace(t_min - 0.1, t_max + 0.1, num_points)

    tent_values = np.zeros((len(finite), num_points))
    for i, (b, d) in enumerate(zip(births, deaths)):
        mid = (b + d) / 2
        for j, t in enumerate(t_grid):
            if b <= t <= mid:
                tent_values[i, j] = t - b
            elif mid < t <= d:
                tent_values[i, j] = d - t

    landscapes = np.zeros((k_max, num_points))
    for j in range(num_points):
        sorted_vals = np.sort(tent_values[:, j])[::-1]
        for k in range(min(k_max, len(sorted_vals))):
            landscapes[k, j] = sorted_vals[k]

    return landscapes, t_grid


landscapes_h1, t_grid = persistence_landscape(dgms[1], k_max=3)

fig3, ax3 = plt.subplots(1, 1, figsize=(10, 4))
colors_land = ['indianred', 'steelblue', 'forestgreen']
labels = [r'$\lambda_1$', r'$\lambda_2$', r'$\lambda_3$']
for k in range(3):
    if np.max(landscapes_h1[k]) > 1e-6:
        ax3.fill_between(t_grid, landscapes_h1[k], alpha=0.15, color=colors_land[k])
        ax3.plot(t_grid, landscapes_h1[k], color=colors_land[k],
                 linewidth=2, label=labels[k])

ax3.set_xlabel('$t$', fontsize=13)
ax3.set_ylabel('$\\lambda_k(t)$', fontsize=13)
ax3.set_title('Paisaje de persistencia ($H_1$)', fontsize=13)
ax3.legend(fontsize=12)

plt.tight_layout()
plt.savefig('../../memoria/figuras/cap2/paisaje_persistencia.png', dpi=200, bbox_inches='tight')
plt.savefig('../../memoria/figuras/cap2/paisaje_persistencia.pdf', bbox_inches='tight')
print("Paisaje de persistencia guardado.")

print("\nTodas las figuras generadas en memoria/figuras/cap2/")
