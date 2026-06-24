"""
Figura 4.X: Pipeline de HalluZig - Filtración zigzag sobre grafos de atención.
Genera una figura con dos filas:
  Fila superior: secuencia de 5 grafos de atención (G1...G5) con grafos unión intercalados
                 y flechas zigzag
  Fila inferior: barcode zigzag resultante (H0 y H1)

Datos sintéticos diseñados para mostrar evolución topológica realista.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

np.random.seed(123)

# --- Configuración ---
T = 7  # tokens
n_layers = 5  # capas (grafos G1..G5)
tokens = [f'$t_{i+1}$' for i in range(T)]

# Posiciones fijas para todos los grafos (circular)
angles = np.linspace(0, 2*np.pi, T, endpoint=False) + np.pi/2
base_pos = {i: (np.cos(a), np.sin(a)) for i, a in enumerate(angles)}

# --- Generar grafos sintéticos con evolución topológica ---
# Diseñados para que:
# - G1: componentes separadas, sin ciclos
# - G2: se forma un ciclo, se conectan componentes
# - G3: el ciclo persiste, aparece otro
# - G4: un ciclo desaparece, el otro persiste
# - G5: todo conectado, un ciclo persiste

edges_per_layer = [
    # G1: dos componentes {0,1,2} y {3,4,5,6}, sin ciclos
    [(0,1), (1,2), (3,4), (4,5), (5,6)],
    # G2: se conectan las componentes, aparece ciclo 0-1-2-0
    [(0,1), (1,2), (2,0), (2,3), (4,5), (5,6)],
    # G3: ciclo 0-1-2 persiste, aparece ciclo 3-4-5-3
    [(0,1), (1,2), (2,0), (2,3), (3,4), (4,5), (5,3)],
    # G4: ciclo 0-1-2 desaparece (pierde arista 2-0), ciclo 3-4-5 persiste
    [(0,1), (1,2), (2,3), (3,4), (4,5), (5,3), (5,6)],
    # G5: ciclo 3-4-5 persiste, nueva conexión 6-0
    [(0,1), (1,2), (2,3), (3,4), (4,5), (5,3), (5,6), (6,0)],
]

# Colores para aristas nuevas vs persistentes
def get_edge_sets(layer_idx):
    """Devuelve edges del grafo actual"""
    return set(tuple(sorted(e)) for e in edges_per_layer[layer_idx])

# --- Crear figura ---
fig = plt.figure(figsize=(16, 8))

# Layout: fila superior para grafos, fila inferior para barcode
gs = fig.add_gridspec(2, 1, height_ratios=[1.6, 1], hspace=0.35)
gs_top = gs[0].subgridspec(1, 1)
gs_bot = gs[1].subgridspec(1, 1)

ax_graphs = fig.add_subplot(gs_top[0])
ax_barcode = fig.add_subplot(gs_bot[0])

# ============================================================
# FILA SUPERIOR: Grafos con flechas zigzag
# ============================================================
graph_spacing = 2.8
scale = 0.45
y_offset = 0.0

for layer_idx in range(n_layers):
    x_center = layer_idx * graph_spacing
    sub_pos = {i: (x * scale + x_center, y * scale + y_offset)
               for i, (x, y) in base_pos.items()}

    edges = edges_per_layer[layer_idx]
    edge_set = get_edge_sets(layer_idx)

    # Determinar aristas nuevas vs heredadas
    if layer_idx > 0:
        prev_set = get_edge_sets(layer_idx - 1)
        new_edges = edge_set - prev_set
        kept_edges = edge_set & prev_set
        lost_from_prev = prev_set - edge_set
    else:
        new_edges = edge_set
        kept_edges = set()

    # Dibujar aristas
    for e in edges:
        i, j = e
        x1, y1 = sub_pos[i]
        x2, y2 = sub_pos[j]
        e_sorted = tuple(sorted(e))
        if e_sorted in new_edges and layer_idx > 0:
            # Arista nueva: azul
            ax_graphs.plot([x1, x2], [y1, y2], '-', color='#2196F3',
                          alpha=0.7, linewidth=2.0, zorder=1)
        else:
            # Arista persistente o inicial: negro
            ax_graphs.plot([x1, x2], [y1, y2], 'k-',
                          alpha=0.6, linewidth=1.5, zorder=1)

    # Dibujar nodos
    for i in range(T):
        circle = plt.Circle(sub_pos[i], 0.13, color='steelblue', alpha=0.9, zorder=3)
        ax_graphs.add_patch(circle)
        ax_graphs.text(sub_pos[i][0], sub_pos[i][1], tokens[i], ha='center', va='center',
                      fontsize=5, color='white', fontweight='bold', zorder=4)

    # Etiqueta del grafo
    ax_graphs.text(x_center, y_offset - 0.85,
                  f'$G_{{{layer_idx+1}}}$', ha='center', fontsize=11, fontweight='bold')

    # Info topológica debajo
    # Calcular H0 y H1 manualmente
    G = nx.Graph()
    G.add_nodes_from(range(T))
    G.add_edges_from(edges)
    n_components = nx.number_connected_components(G)
    n_cycles = len(edges) - (T - n_components)  # e - v + c
    ax_graphs.text(x_center, y_offset - 1.1,
                  f'$\\beta_0={n_components},\\ \\beta_1={n_cycles}$',
                  ha='center', fontsize=7, color='gray')

    # Flechas zigzag entre grafos
    if layer_idx < n_layers - 1:
        arrow_x = x_center + graph_spacing / 2
        # Flecha derecha (hookrightarrow)
        if layer_idx % 2 == 0:
            ax_graphs.annotate('', xy=(arrow_x + 0.3, y_offset),
                             xytext=(arrow_x - 0.3, y_offset),
                             arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
            ax_graphs.text(arrow_x, y_offset + 0.2, '$\\hookrightarrow$',
                         ha='center', fontsize=10, color='#333')
        else:
            ax_graphs.annotate('', xy=(arrow_x - 0.3, y_offset),
                             xytext=(arrow_x + 0.3, y_offset),
                             arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
            ax_graphs.text(arrow_x, y_offset + 0.2, '$\\hookleftarrow$',
                         ha='center', fontsize=10, color='#333')

ax_graphs.set_xlim(-1.0, (n_layers - 1) * graph_spacing + 1.0)
ax_graphs.set_ylim(-1.5, 1.0)
ax_graphs.set_aspect('equal')
ax_graphs.axis('off')
ax_graphs.set_title('Evolución de grafos de atención a lo largo de las capas',
                    fontsize=12, pad=10)

# ============================================================
# FILA INFERIOR: Barcode zigzag
# ============================================================
# Barcode sintético coherente con los grafos diseñados
# H0: componentes que se fusionan
# H1: ciclos que aparecen y desaparecen

colors_dim = ['#2196F3', '#F44336']

# H0 intervals: [birth_layer, death_layer]
# Al inicio hay 2 componentes, se fusionan en G2
h0_bars = [
    (0, 5, '$H_0$: comp. principal'),  # componente que persiste
    (0, 1.5, '$H_0$: comp. $\\{t_4,...,t_7\\}$'),  # se fusiona en G2
]

# H1 intervals:
h1_bars = [
    (1, 3.5, '$H_1$: ciclo $t_1$-$t_2$-$t_3$'),      # ciclo 0-1-2, nace en G2, muere en G4
    (2, 5, '$H_1$: ciclo $t_4$-$t_5$-$t_6$'),         # ciclo 3-4-5, nace en G3, persiste
]

all_bars = [(b, d, lbl, 0) for b, d, lbl in h0_bars] + \
           [(b, d, lbl, 1) for b, d, lbl in h1_bars]

bar_idx = 0
yticks = []
ylabels = []
for birth, death, label, dim in all_bars:
    color = colors_dim[dim]
    if death >= 5:
        # Persiste hasta el final
        ax_barcode.barh(bar_idx, death - birth, left=birth, height=0.6,
                       color=color, alpha=0.7, edgecolor='k', linewidth=0.5)
        ax_barcode.plot(death, bar_idx, '>', color=color, markersize=6)
    else:
        ax_barcode.barh(bar_idx, death - birth, left=birth, height=0.6,
                       color=color, alpha=0.7, edgecolor='k', linewidth=0.5)
    yticks.append(bar_idx)
    ylabels.append(label)
    bar_idx += 1

# Líneas verticales para las capas
for l in range(n_layers):
    ax_barcode.axvline(x=l, color='gray', alpha=0.2, linestyle='--', linewidth=0.8)
    ax_barcode.text(l, bar_idx + 0.3, f'$G_{{{l+1}}}$', ha='center', fontsize=9, color='gray')

ax_barcode.set_xlabel('Paso de la filtración zigzag (capas del modelo)', fontsize=10)
ax_barcode.set_yticks(yticks)
ax_barcode.set_yticklabels(ylabels, fontsize=8)
ax_barcode.set_xlim(-0.3, 5.5)
ax_barcode.set_ylim(-0.8, bar_idx + 0.8)
ax_barcode.set_title('Barcode zigzag resultante', fontsize=12, pad=8)

# Leyenda
h0_patch = mpatches.Patch(color=colors_dim[0], alpha=0.7, label='$H_0$ (componentes)')
h1_patch = mpatches.Patch(color=colors_dim[1], alpha=0.7, label='$H_1$ (ciclos)')
ax_barcode.legend(handles=[h0_patch, h1_patch], fontsize=9, loc='upper right')
ax_barcode.grid(True, alpha=0.1, axis='x')

plt.tight_layout()

# Guardar
out_dir = '/Users/alejandromorales/Library/CloudStorage/OneDrive-Personal/Documentos/Máster en Inteligencia Artificial/2º SEMESTRE/Trabajo Fin de Máster/memoria/figuras/cap4'
plt.savefig(f'{out_dir}/pipeline_halluzig.pdf', bbox_inches='tight', dpi=200)
plt.savefig(f'{out_dir}/pipeline_halluzig.png', bbox_inches='tight', dpi=200)
plt.close()
print("Figura pipeline HalluZig guardada")
