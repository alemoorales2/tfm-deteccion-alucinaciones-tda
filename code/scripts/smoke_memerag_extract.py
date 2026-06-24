"""smoke_memerag_extract.py - MICRO-BENCHMARK de extracción RAG (Fase 5).

Mide el coste real por muestra de la etapa CON modelo sobre secuencias RAG largas
(contexto ~1000 tokens), que es la gran incógnita del experimento RAG: a T grande,
el cross-barcode de TOHA y el zigzag de HalluZig escalan mal. Carga unas pocas
muestras del smoke de generación y cronometra POR SEPARADO el forward y cada método,
a varios T_max, para decidir si A2 (2 modelos) es viable o si hay que restringir a Llama.

    .venv/bin/python code/scripts/smoke_memerag_extract.py

No persiste features; solo imprime tiempos y shapes. No es parte del pipeline.
"""

import os
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
import torch

from tfm_hallu import config
from tfm_hallu.data.base import Sample
from tfm_hallu.extraction import encode_sample, extract_attention, load_model
from tfm_hallu.methods import (
    sample_halluzig_features,
    sample_head_divergences,
    sample_lap_eigenvalues,
)

RAW = config.BENCHMARKS_DIR / "memerag-es" / "llama32-3b_smoke_raw.parquet"
N_SAMPLES = 2
T_MAX_GRID = [512, 1024, 1536]


def main() -> int:
    df = pd.read_parquet(RAW)
    # ordena por longitud de contexto desc para coger casos largos (peor caso)
    df = df.sort_values("n_ctx_tok", ascending=False).head(N_SAMPLES).reset_index(drop=True)
    samples = [
        Sample(id=str(r["query_id"]), prompt=str(r["prompt"]), respuesta=str(r["generated"]),
               label=0, lang="es", source="rag", meta={"chat": "plain"})
        for _, r in df.iterrows()
    ]
    print(f"Muestras: {[s.id for s in samples]}  ctx_tok={df.n_ctx_tok.tolist()}", flush=True)

    spec = config.MODELS["llama32-3b"]
    model, tok, device = load_model(spec)
    tcfg = getattr(model.config, "text_config", model.config)
    L, H, k = tcfg.num_hidden_layers, tcfg.num_attention_heads, config.LAPEIGVALS_K
    print(f"device={device}  L={L}  H={H}", flush=True)

    def timed(fn):
        t0 = time.time()
        out = fn()
        return out, time.time() - t0

    for t_max in T_MAX_GRID:
        print(f"\n===== T_max={t_max} =====", flush=True)
        for s in samples:
            enc = encode_sample(tok, s, t_max)
            T = enc.n_prompt + enc.n_tok_resp
            (attn, hidden), t_fwd = timed(lambda: extract_attention(model, enc, device))
            # attn esperado [L,H,T,T]
            ashape = tuple(attn.shape) if hasattr(attn, "shape") else "?"
            _, t_toha = timed(lambda: sample_head_divergences(attn, enc.n_prompt))
            _, t_lap = timed(lambda: sample_lap_eigenvalues(attn, k))
            _, t_hz = timed(lambda: sample_halluzig_features(attn))
            del attn
            if device == "mps":
                torch.mps.empty_cache()
            total = t_fwd + t_toha + t_lap + t_hz
            print(
                f"  T={T:5d} (P={enc.n_prompt},R={enc.n_tok_resp},trunc={enc.truncated})  "
                f"attn={ashape}  | fwd={t_fwd:6.2f}s  toha={t_toha:6.2f}s  "
                f"lap={t_lap:5.2f}s  HALLUZIG={t_hz:7.2f}s  -> {total:6.2f}s/muestra",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
