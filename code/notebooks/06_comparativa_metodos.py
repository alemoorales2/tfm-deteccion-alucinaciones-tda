"""
Figura 4.X: Jerarquía de expresividad - LapEigvals vs TOHA vs HalluZig.
Diagrama esquemático que muestra qué información extrae cada método
del mismo grafo de atención.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import networkx as nx

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# --- Configuración común ---
T = 6
angles = np.linspace(0, 2*np.pi, T, endpoint=False) + np.pi/2
base_pos = {i: (np.cos(a), np.sin(a)) for i, a in enumerate(angles)}

edges = [(0,1), (1,2), (2,0), (2,3), (3,4), (4,5)]
node_color = 'steelblue'

def draw_small_graph(ax, pos, edges, x_off, y_off, scale=0.3, alpha=0.6):
    """Dibuja un mini-grafo"""
    sp = {i: (x*scale + x_off, y*scale + y_off) for i, (x,y) in pos.items()}
    for i, j in edges:
        ax.plot([sp[i][0], sp[j][0]], [sp[i][1], sp[j][1]],
               'k-', alpha=alpha, linewidth=1.0, zorder=1)
    for i in range(T):
        circle = plt.Circle(sp[i], 0.06, color=node_color, alpha=0.8, zorder=3)
        ax.add_patch(circle)

# ================================================================
# Panel 1: LapEigvals
# ================================================================
ax = axes[0]
ax.set_title('LapEigvals\n(espectral)', fontsize=12, fontweight='bold', pad=12)

# Grafo de entrada
draw_small_graph(ax, base_pos, edges, 0, 1.8, scale=0.4)
ax.text(0, 1.15, 'Grafo $G^{(l,h)}$', ha='center', fontsize=9, color='gray')

# Flecha
ax.annotate('', xy=(0, 0.6), xytext=(0, 1.0),
           arrowprops=dict(arrowstyle='->', color='#333', lw=2))
ax.text(0.55, 0.8, 'Laplaciano\n$L = D - A$', ha='center', fontsize=8, color='#555')

# Resultado: vector de autovalores
eigenvals = [0, 0.38, 1.0, 1.62, 2.0, 3.0]
bar_y = np.linspace(-0.2, 0.4, len(eigenvals))
ax.barh(bar_y, eigenvals, height=0.08, color='#9C27B0', alpha=0.7, edgecolor='k', linewidth=0.3)
ax.text(0, -0.55, 'Vector de autovalores $\\lambda_1, \\dots, \\lambda_n$',
       ha='center', fontsize=9, color='#9C27B0')

ax.set_xlim(-1.2, 3.5)
ax.set_ylim(-0.9, 2.5)
ax.set_aspect('equal')
ax.axis('off')

# ================================================================
# Panel 2: TOHA
# ================================================================
ax = axes[1]
ax.set_title('TOHA\n(PH estándar, $H_0$)', fontsize=12, fontweight='bold', pad=12)

# Grafo con partición P/R
sp = {i: (x*0.4, y*0.4 + 1.8) for i, (x,y) in base_pos.items()}
# Prompt = {0,1,2}, Response = {3,4,5}
for i, j in edges:
    ax.plot([sp[i][0], sp[j][0]], [sp[i][1], sp[j][1]],
           'k-', alpha=0.6, linewidth=1.0, zorder=1)
for i in range(T):
    color = '#4CAF50' if i < 3 else '#FF9800'
    circle = plt.Circle(sp[i], 0.07, color=color, alpha=0.85, zorder=3)
    ax.add_patch(circle)
ax.text(-0.5, 1.25, '$P$', ha='center', fontsize=9, color='#4CAF50', fontweight='bold')
ax.text(0.5, 1.25, '$R$', ha='center', fontsize=9, color='#FF9800', fontweight='bold')
ax.text(0, 1.05, 'Partición prompt/respuesta', ha='center', fontsize=8, color='gray')

# Flecha
ax.annotate('', xy=(0, 0.55), xytext=(0, 0.9),
           arrowprops=dict(arrowstyle='->', color='#333', lw=2))
ax.text(0.8, 0.7, 'Filtración VR\n$d = 1 - A$', ha='center', fontsize=8, color='#555')

# Diagrama de persistencia H0
np.random.seed(77)
births = np.array([0, 0, 0, 0, 0])
deaths = np.array([0.1, 0.25, 0.4, 0.55, np.inf])
for b, d in zip(births, deaths):
    if d == np.inf:
        ax.scatter(b, 0.3, c='#2196F3', s=60, marker='^', zorder=3,
                  edgecolors='k', linewidths=0.5)
    else:
        ax.scatter(b, d - 0.1, c='#2196F3', s=40, zorder=3,
                  edgecolors='k', linewidths=0.5)
ax.plot([-0.1, 0.7], [-0.35, 0.35], 'k--', alpha=0.2, linewidth=0.8)
ax.text(0, -0.55, 'Diagrama $H_0$ + MTop-Div',
       ha='center', fontsize=9, color='#2196F3')

ax.set_xlim(-1.2, 1.8)
ax.set_ylim(-0.9, 2.5)
ax.set_aspect('equal')
ax.axis('off')

# ================================================================
# Panel 3: HalluZig
# ================================================================
ax = axes[2]
ax.set_title('HalluZig\n(PH zigzag, $H_0 + H_1$)', fontsize=12, fontweight='bold', pad=12)

# Secuencia de mini-grafos (3 capas)
layers_edges = [
    [(0,1), (1,2), (2,3), (3,4)],
    [(0,1), (1,2), (2,0), (2,3), (4,5)],
    [(1,2), (2,0), (3,4), (4,5), (5,3)],
]
for k, le in enumerate(layers_edges):
    x_off = (k - 1) * 1.0
    draw_small_graph(ax, base_pos, le, x_off, 1.8, scale=0.25, alpha=0.5)
    ax.text(x_off, 1.35, f'$G_{{{k+1}}}$', ha='center', fontsize=8, color='gray')
    if k < 2:
        ax.annotate('', xy=(x_off + 0.35, 1.8), xytext=(x_off + 0.65, 1.8),
                   arrowprops=dict(arrowstyle='<-', color='#333', lw=1))

ax.text(0, 1.1, 'Filtración zigzag entre capas', ha='center', fontsize=8, color='gray')

# Flecha
ax.annotate('', xy=(0, 0.55), xytext=(0, 0.9),
           arrowprops=dict(arrowstyle='->', color='#333', lw=2))
ax.text(0.9, 0.7, 'Zigzag PH\n$H_0 + H_1$', ha='center', fontsize=8, color='#555')

# Barcode zigzag (mini)
bars_h0 = [(0, 2.8, '#2196F3'), (0, 1.2, '#2196F3')]
bars_h1 = [(0.8, 2.5, '#F44336'), (1.5, 2.8, '#F44336')]
all_bars = bars_h0 + bars_h1
for idx, (b, d, c) in enumerate(all_bars):
    y_bar = idx * 0.18 - 0.3
    ax.barh(y_bar, d - b, left=b - 1.2, height=0.14,
           color=c, alpha=0.7, edgecolor='k', linewidth=0.3)

ax.text(0, -0.55, 'Barcode zigzag + vectorización',
       ha='center', fontsize=9, color='#F44336')

ax.set_xlim(-1.8, 2.2)
ax.set_ylim(-0.9, 2.5)
ax.set_aspect('equal')
ax.axis('off')

# ================================================================
# Flechas de jerarquía entre paneles
# ================================================================
# Textos de jerarquía
fig.text(0.35, 0.03, '$\\longrightarrow$ Expresividad creciente $\\longrightarrow$',
        ha='center', fontsize=11, color='gray', style='italic',
        transform=fig.transFigure)

plt.tight_layout(rect=[0, 0.05, 1, 1])

out_dir = '/Users/alejandromorales/Library/CloudStorage/OneDrive-Personal/Documentos/Máster en Inteligencia Artificial/2º SEMESTRE/Trabajo Fin de Máster/memoria/figuras/cap4'
plt.savefig(f'{out_dir}/comparativa_metodos.pdf', bbox_inches='tight', dpi=200)
plt.savefig(f'{out_dir}/comparativa_metodos.png', bbox_inches='tight', dpi=200)
plt.close()
print("Figura comparativa guardada")
