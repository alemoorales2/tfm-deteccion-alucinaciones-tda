"""
Figura 4.1: De la matriz de atención al grafo ponderado.
Genera una figura con 3 paneles:
  (a) Matriz de atención causal 6x6 con mapa de calor
  (b) Grafo ponderado correspondiente (grosor = peso de atención)
  (c) Filtración VR a dos umbrales de distancia (ε bajo y ε alto)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Reproducibilidad
np.random.seed(42)

# --- Matriz de atención causal sintética (6 tokens) ---
tokens = ['El', 'gato', 'se', 'sentó', 'en', 'la']
T = len(tokens)

# Crear una matriz causal realista (triangular inferior + diagonal)
A = np.zeros((T, T))
# Diagonal: cada token se atiende a sí mismo
np.fill_diagonal(A, 0.3)
# Llenar con valores que simulen patrones de atención
raw = np.array([
    [1.0,  0.0,  0.0,  0.0,  0.0,  0.0],   # 'El' solo se atiende a sí mismo
    [0.25, 0.75, 0.0,  0.0,  0.0,  0.0],   # 'gato' atiende mucho a sí mismo, algo a 'El'
    [0.10, 0.15, 0.75, 0.0,  0.0,  0.0],   # 'se' atiende poco a anteriores
    [0.05, 0.55, 0.05, 0.35, 0.0,  0.0],   # 'sentó' atiende mucho a 'gato'
    [0.05, 0.10, 0.10, 0.45, 0.30, 0.0],   # 'en' atiende a 'sentó'
    [0.05, 0.05, 0.05, 0.10, 0.35, 0.40],  # 'la' atiende a 'en' y a sí misma
])
# Normalizar filas para que sumen 1
A = raw / raw.sum(axis=1, keepdims=True)

# --- Posiciones de nodos para los grafos ---
# Disposición circular
angles = np.linspace(0, 2*np.pi, T, endpoint=False) + np.pi/2
pos = {i: (np.cos(a), np.sin(a)) for i, a in enumerate(angles)}

# --- Crear figura ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

# --- Panel (a): Matriz de atención ---
ax = axes[0]
im = ax.imshow(A, cmap='Blues', vmin=0, vmax=0.8, aspect='equal')
ax.set_xticks(range(T))
ax.set_xticklabels(tokens, fontsize=8)
ax.set_yticks(range(T))
ax.set_yticklabels(tokens, fontsize=8)
ax.set_xlabel('Clave ($j$)', fontsize=9)
ax.set_ylabel('Consulta ($i$)', fontsize=9)
ax.set_title('(a) Matriz de atención $A$', fontsize=10, pad=8)
# Añadir valores en las celdas
for i in range(T):
    for j in range(T):
        if A[i, j] > 0.01:
            color = 'white' if A[i, j] > 0.4 else 'black'
            ax.text(j, i, f'{A[i,j]:.2f}', ha='center', va='center',
                   fontsize=6.5, color=color)
        else:
            ax.text(j, i, '0', ha='center', va='center',
                   fontsize=6.5, color='gray')

# Colorbar
cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cbar.ax.tick_params(labelsize=7)

# --- Panel (b): Grafo ponderado ---
ax = axes[1]
G = nx.DiGraph()
for i in range(T):
    G.add_node(i)

# Añadir aristas con peso > umbral mínimo
threshold = 0.08
edges = []
widths = []
colors_edges = []
for i in range(T):
    for j in range(T):
        if A[i, j] > threshold and i != j:
            G.add_edge(j, i, weight=A[i, j])  # j -> i (j es atendido por i)
            edges.append((j, i))
            widths.append(A[i, j] * 5)
            colors_edges.append(A[i, j])

# Dibujar nodos
nx.draw_networkx_nodes(G, pos, ax=ax, node_color='steelblue',
                       node_size=400, alpha=0.9)
nx.draw_networkx_labels(G, pos, ax=ax,
                        labels={i: t for i, t in enumerate(tokens)},
                        font_size=8, font_color='white', font_weight='bold')

# Dibujar aristas con grosor proporcional al peso
if edges:
    nx.draw_networkx_edges(G, pos, edgelist=edges, ax=ax,
                          width=widths, alpha=0.6,
                          edge_color=colors_edges, edge_cmap=plt.cm.Blues,
                          edge_vmin=0, edge_vmax=0.8,
                          arrows=True, arrowsize=10,
                          connectionstyle='arc3,rad=0.1')

ax.set_title('(b) Grafo de atención $G$', fontsize=10, pad=8)
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
ax.axis('off')

# --- Panel (c): Filtración VR a dos umbrales ---
ax = axes[2]

# Convertir a distancias
D = 1 - A

# Dos umbrales
eps_low = 0.55   # Solo aristas con atención > 0.45
eps_high = 0.85  # Aristas con atención > 0.15

# Grafo no dirigido simétrico para la filtración
# Simetrizar: d(i,j) = min(D[i,j], D[j,i]) si ambos > 0
D_sym = np.full((T, T), np.inf)
for i in range(T):
    for j in range(i+1, T):
        d_ij = D[i, j] if A[i, j] > 0.01 else np.inf
        d_ji = D[j, i] if A[j, i] > 0.01 else np.inf
        D_sym[i, j] = D_sym[j, i] = min(d_ij, d_ji)

# Dibujar en dos mitades: izquierda ε bajo, derecha ε alto
# Usamos un offset para separar los dos grafos
offset_x = -1.3
offset_y = 0

# Posiciones desplazadas para cada sub-panel
pos_left = {i: (x * 0.7 + offset_x, y * 0.7 + 0.1) for i, (x, y) in pos.items()}
pos_right = {i: (x * 0.7 - offset_x, y * 0.7 + 0.1) for i, (x, y) in pos.items()}

for sub_pos, eps, label, x_label in [(pos_left, eps_low, f'$\\varepsilon = {eps_low}$', offset_x),
                                       (pos_right, eps_high, f'$\\varepsilon = {eps_high}$', -offset_x)]:
    # Nodos
    for i in range(T):
        circle = plt.Circle(sub_pos[i], 0.12, color='steelblue', alpha=0.9, zorder=3)
        ax.add_patch(circle)
        ax.text(sub_pos[i][0], sub_pos[i][1], tokens[i], ha='center', va='center',
               fontsize=6, color='white', fontweight='bold', zorder=4)

    # Aristas que pasan el umbral
    for i in range(T):
        for j in range(i+1, T):
            if D_sym[i, j] <= eps:
                x1, y1 = sub_pos[i]
                x2, y2 = sub_pos[j]
                alpha_val = 0.7 if D_sym[i, j] <= eps_low else 0.35
                lw = 2.0 if D_sym[i, j] <= eps_low else 1.0
                ax.plot([x1, x2], [y1, y2], 'k-', alpha=alpha_val,
                       linewidth=lw, zorder=1)

    ax.text(x_label, -0.85, label, ha='center', fontsize=9)

ax.set_title('(c) Filtración: $d(t_i, t_j) = 1 - A_{ij}$', fontsize=10, pad=8)
ax.set_xlim(-2.3, 2.3)
ax.set_ylim(-1.1, 1.2)
ax.set_aspect('equal')
ax.axis('off')

# Flecha entre los dos sub-grafos
ax.annotate('', xy=(0.15, 0.1), xytext=(-0.15, 0.1),
           arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
ax.text(0, -0.1, '$\\varepsilon \\uparrow$', ha='center', fontsize=9, color='gray')

plt.tight_layout(w_pad=1.5)

# Guardar
out_dir = '/Users/alejandromorales/Library/CloudStorage/OneDrive-Personal/Documentos/Máster en Inteligencia Artificial/2º SEMESTRE/Trabajo Fin de Máster/memoria/figuras/cap4'
plt.savefig(f'{out_dir}/atencion_a_grafo.pdf', bbox_inches='tight', dpi=200)
plt.savefig(f'{out_dir}/atencion_a_grafo.png', bbox_inches='tight', dpi=200)
print("Figuras guardadas en", out_dir)
