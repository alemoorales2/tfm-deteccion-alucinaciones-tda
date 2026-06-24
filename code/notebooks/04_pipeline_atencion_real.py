"""
Ejemplo 4.1: Pipeline completo de la matriz de atención a la homología persistente.
Genera 3 figuras separadas para intercalar con texto explicativo:
  Fig 1: (a) Matriz de atención + (b) Grafo de atención
  Fig 2: (c) Matriz de distancias + (d) Filtración VR a 3 umbrales
  Fig 3: (e) Diagrama de persistencia + (f) Barcode

Datos: GPT-2 (124M params), capa 5, cabeza 3, frase "The cat sat on the mat"
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from ripser import ripser

# --- Cargar modelo y extraer atención ---
print("Cargando GPT-2...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2", attn_implementation="eager")
model.eval()

prompt = "The cat sat on the mat"
inputs = tokenizer(prompt, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
tokens_clean = [t.replace("Ġ", " ").strip() for t in tokens]
T = len(tokens_clean)
print(f"Tokens ({T}): {tokens_clean}")

with torch.no_grad():
    outputs = model(**inputs, output_attentions=True)

layer_idx, head_idx = 5, 3
A = outputs.attentions[layer_idx][0, head_idx].numpy()
print(f"Atención extraída: capa {layer_idx}, cabeza {head_idx}")

# --- Preparar distancias ---
D = 1.0 - A
D_sym = np.full((T, T), 0.0)
for i in range(T):
    for j in range(i+1, T):
        d_ij = D[i, j] if A[i, j] > 0.01 else 1.0
        d_ji = D[j, i] if A[j, i] > 0.01 else 1.0
        D_sym[i, j] = D_sym[j, i] = min(d_ij, d_ji)

# --- Homología persistente ---
result = ripser(D_sym, maxdim=1, distance_matrix=True)
dgms = result['dgms']

# --- Posiciones de nodos (circular) ---
angles = np.linspace(0, 2*np.pi, T, endpoint=False) + np.pi/2
pos = {i: (np.cos(a), np.sin(a)) for i, a in enumerate(angles)}

out_dir = '/Users/alejandromorales/Library/CloudStorage/OneDrive-Personal/Documentos/Máster en Inteligencia Artificial/2º SEMESTRE/Trabajo Fin de Máster/memoria/figuras/cap4'

# =====================================================================
# FIGURA 1: Matriz de atención + Grafo de atención
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# (a) Matriz de atención
ax = axes[0]
im = ax.imshow(A, cmap='Blues', vmin=0, vmax=np.max(A)*1.1, aspect='equal')
ax.set_xticks(range(T))
ax.set_xticklabels(tokens_clean, fontsize=9, rotation=45, ha='right')
ax.set_yticks(range(T))
ax.set_yticklabels(tokens_clean, fontsize=9)
ax.set_xlabel('Clave ($j$)', fontsize=10)
ax.set_ylabel('Consulta ($i$)', fontsize=10)
ax.set_title('(a) Matriz de atención $A^{(5,3)}$', fontsize=11, pad=8)
for i in range(T):
    for j in range(T):
        val = A[i, j]
        if val > 0.01:
            color = 'white' if val > 0.3 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)
        else:
            ax.text(j, i, '0', ha='center', va='center', fontsize=7, color='lightgray')
cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
cbar.ax.tick_params(labelsize=8)

# (b) Grafo de atención
ax = axes[1]
G = nx.DiGraph()
for i in range(T):
    G.add_node(i)

threshold_vis = 0.05
edges, widths, colors_e = [], [], []
for i in range(T):
    for j in range(T):
        if A[i, j] > threshold_vis and i != j:
            edges.append((j, i))
            widths.append(A[i, j] * 7)
            colors_e.append(A[i, j])
            G.add_edge(j, i)

nx.draw_networkx_nodes(G, pos, ax=ax, node_color='steelblue', node_size=500, alpha=0.9)
nx.draw_networkx_labels(G, pos, ax=ax,
                        labels={i: t for i, t in enumerate(tokens_clean)},
                        font_size=9, font_color='white', font_weight='bold')
if edges:
    nx.draw_networkx_edges(G, pos, edgelist=edges, ax=ax,
                          width=widths, alpha=0.5,
                          edge_color=colors_e, edge_cmap=plt.cm.Blues,
                          edge_vmin=0, edge_vmax=max(colors_e),
                          arrows=True, arrowsize=10,
                          connectionstyle='arc3,rad=0.12')
ax.set_title('(b) Grafo de atención $G^{(5,3)}$', fontsize=11, pad=8)
ax.set_xlim(-1.7, 1.7)
ax.set_ylim(-1.7, 1.7)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout(w_pad=2)
plt.savefig(f'{out_dir}/ejemplo_paso1.pdf', bbox_inches='tight', dpi=200)
plt.savefig(f'{out_dir}/ejemplo_paso1.png', bbox_inches='tight', dpi=200)
plt.close()
print("Figura 1 guardada (paso 1: atención + grafo)")

# =====================================================================
# FIGURA 2: Matriz de distancias + Filtración VR
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# (c) Matriz de distancias simetrizada
ax = axes[0]
# Poner diagonal en blanco
D_display = D_sym.copy()
np.fill_diagonal(D_display, np.nan)
im2 = ax.imshow(D_display, cmap='YlOrRd', vmin=0, vmax=1, aspect='equal')
ax.set_xticks(range(T))
ax.set_xticklabels(tokens_clean, fontsize=9, rotation=45, ha='right')
ax.set_yticks(range(T))
ax.set_yticklabels(tokens_clean, fontsize=9)
ax.set_title('(c) Distancias $d_{ij} = 1 - A_{ij}$ (simetrizada)', fontsize=11, pad=8)
for i in range(T):
    for j in range(T):
        if i != j:
            val = D_sym[i, j]
            color = 'white' if val > 0.65 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)
cbar2 = fig.colorbar(im2, ax=ax, shrink=0.85, pad=0.03)
cbar2.ax.tick_params(labelsize=8)

# (d) Filtración VR a 3 umbrales
ax = axes[1]
epsilons = [0.4, 0.7, 0.95]
n_eps = len(epsilons)
spacing = 3.0

for k, eps in enumerate(epsilons):
    offset_x = (k - (n_eps-1)/2) * spacing
    sub_pos = {i: (x * 0.6 + offset_x, y * 0.6) for i, (x, y) in pos.items()}

    for i in range(T):
        circle = plt.Circle(sub_pos[i], 0.14, color='steelblue', alpha=0.85, zorder=3)
        ax.add_patch(circle)
        ax.text(sub_pos[i][0], sub_pos[i][1], tokens_clean[i], ha='center', va='center',
               fontsize=5.5, color='white', fontweight='bold', zorder=4)

    n_edges = 0
    for i in range(T):
        for j in range(i+1, T):
            if D_sym[i, j] <= eps and D_sym[i, j] > 0:
                x1, y1 = sub_pos[i]
                x2, y2 = sub_pos[j]
                alpha_val = max(0.25, 1.0 - D_sym[i, j])
                ax.plot([x1, x2], [y1, y2], 'k-', alpha=alpha_val, linewidth=1.5, zorder=1)
                n_edges += 1

    ax.text(offset_x, -1.15, f'$\\varepsilon = {eps}$\n({n_edges} aristas)',
           ha='center', fontsize=8)

    if k < n_eps - 1:
        arr_x = offset_x + spacing/2
        ax.annotate('', xy=(arr_x - 0.15, 0), xytext=(arr_x - 0.65, 0),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

ax.set_title('(d) Filtración de Vietoris-Rips', fontsize=11, pad=8)
ax.set_xlim(-spacing - 0.8, spacing + 0.8)
ax.set_ylim(-1.6, 1.2)
ax.set_aspect('equal')
ax.axis('off')

plt.tight_layout(w_pad=1.5)
plt.savefig(f'{out_dir}/ejemplo_paso2.pdf', bbox_inches='tight', dpi=200)
plt.savefig(f'{out_dir}/ejemplo_paso2.png', bbox_inches='tight', dpi=200)
plt.close()
print("Figura 2 guardada (paso 2: distancias + filtración)")

# =====================================================================
# FIGURA 3: Diagrama de persistencia + Barcode
# =====================================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

colors_dim = ['#2196F3', '#F44336']
labels_dim = ['$H_0$ (componentes)', '$H_1$ (ciclos)']

# (e) Diagrama de persistencia
ax = axes[0]
max_val = 0
for dim in range(min(2, len(dgms))):
    dgm = dgms[dim]
    finite = dgm[dgm[:, 1] < np.inf]
    if len(finite) > 0:
        ax.scatter(finite[:, 0], finite[:, 1], c=colors_dim[dim], s=50,
                  alpha=0.7, label=labels_dim[dim], zorder=3, edgecolors='k', linewidths=0.5)
        max_val = max(max_val, finite[:, 1].max())
    infinite = dgm[dgm[:, 1] == np.inf]
    if len(infinite) > 0:
        death_inf = max_val * 1.15 if max_val > 0 else 1.0
        ax.scatter(infinite[:, 0], [death_inf] * len(infinite),
                  c=colors_dim[dim], s=70, marker='^', alpha=0.7, zorder=3,
                  edgecolors='k', linewidths=0.5, label=f'{labels_dim[dim]} ($\\infty$)')

lim = max(max_val * 1.25, 0.6) if max_val > 0 else 1.0
ax.plot([0, lim], [0, lim], 'k--', alpha=0.3, linewidth=0.8)
ax.set_xlabel('Nacimiento ($b$)', fontsize=10)
ax.set_ylabel('Muerte ($d$)', fontsize=10)
ax.set_title('(e) Diagrama de persistencia', fontsize=11, pad=8)
ax.legend(fontsize=8, loc='lower right')
ax.set_xlim(-0.02, lim)
ax.set_ylim(-0.02, lim)
ax.set_aspect('equal')
ax.grid(True, alpha=0.15)

# (f) Barcode
ax = axes[1]
bar_idx = 0
max_death = max_val * 1.2 if max_val > 0 else 1.0
for dim in range(min(2, len(dgms))):
    dgm = dgms[dim]
    sorted_idx = np.argsort(dgm[:, 0])
    for idx in sorted_idx:
        birth, death = dgm[idx]
        if death == np.inf:
            death_plot = max_death
            ax.barh(bar_idx, death_plot - birth, left=birth, height=0.7,
                   color=colors_dim[dim], alpha=0.7, edgecolor='k', linewidth=0.3)
            ax.plot(death_plot, bar_idx, '>', color=colors_dim[dim], markersize=5)
        else:
            ax.barh(bar_idx, death - birth, left=birth, height=0.7,
                   color=colors_dim[dim], alpha=0.7, edgecolor='k', linewidth=0.3)
        bar_idx += 1

ax.set_xlabel('Parámetro de filtración $\\varepsilon$', fontsize=10)
ax.set_ylabel('Característica topológica', fontsize=10)
ax.set_title('(f) Código de barras', fontsize=11, pad=8)
h0_patch = mpatches.Patch(color=colors_dim[0], alpha=0.7, label='$H_0$')
h1_patch = mpatches.Patch(color=colors_dim[1], alpha=0.7, label='$H_1$')
ax.legend(handles=[h0_patch, h1_patch], fontsize=9, loc='upper right')
ax.grid(True, alpha=0.15, axis='x')

plt.tight_layout(w_pad=2)
plt.savefig(f'{out_dir}/ejemplo_paso3.pdf', bbox_inches='tight', dpi=200)
plt.savefig(f'{out_dir}/ejemplo_paso3.png', bbox_inches='tight', dpi=200)
plt.close()
print("Figura 3 guardada (paso 3: diagrama + barcode)")

# --- Imprimir datos para el texto ---
print("\n=== DATOS PARA EL TEXTO ===")
print(f"Tokens: {tokens_clean}")
print(f"Matriz A (capa {layer_idx}, cabeza {head_idx}):")
print(np.array2string(A, precision=2, suppress_small=True))
print(f"\nMatriz D simetrizada:")
print(np.array2string(D_sym, precision=2, suppress_small=True))
print(f"\nH0 barcode: {dgms[0]}")
if len(dgms) > 1:
    print(f"H1 barcode: {dgms[1]}")
print(f"\nNúmero de aristas por umbral:")
for eps in [0.4, 0.7, 0.95]:
    n = sum(1 for i in range(T) for j in range(i+1, T) if 0 < D_sym[i,j] <= eps)
    print(f"  eps={eps}: {n} aristas")
