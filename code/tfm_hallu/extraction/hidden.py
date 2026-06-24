"""Estado oculto del último token por capa (etapa CON modelo, para SEPs).

SEPs (Cap. 5 §5.4) entrena una sonda lineal sobre el hidden state del último
token de la respuesta, una por capa. Del forward con `output_hidden_states=True`
se obtiene una tupla de L+1 tensores [1, T, d_model] (embeddings + L bloques); se
toma, por capa, el vector del último token (posición T-1, el último de la
respuesta). Se persiste en float32: los modelos en bfloat16 (Gemma 3) tienen
activaciones outlier de gran magnitud en capas tardías que, casteadas a float16
(máximo ~65504), desbordan a infinito; float32 preserva el rango de bfloat16 y la
sonda lineal entrena sin valores no finitos ([n, L+1, d_model]).
"""

import numpy as np
import torch


@torch.no_grad()
def last_token_hidden(hidden_states) -> np.ndarray:
    """`hidden_states`: tupla de (L+1) tensores [1, T, d_model]. Devuelve un array
    [L+1, d_model] en float32 (CPU) con el último token de cada capa."""
    vecs = [h[0, -1, :] for h in hidden_states]
    return torch.stack(vecs).to(torch.float32).cpu().numpy()
