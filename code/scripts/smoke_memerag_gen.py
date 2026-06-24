"""smoke_memerag_gen.py [N] - SMOKE de reconocimiento del experimento RAG (Fase 5).

Genera respuestas RAG con Llama 3.2 3B sobre MEMERAG-es (preguntas nativas en
español + 5 pasajes recuperados de MIRACL). NO etiqueta ni extrae: solo genera y
cachea, para medir base rate de infidelidad tras el juicio (que se hace aparte con
subagentes). No es parte del pipeline definitivo; es un spike para decidir el
diseño A2 antes de comprometer un run grande.

    .venv/bin/python code/scripts/smoke_memerag_gen.py 40

Lee data/benchmarks/memerag-repo/data/memerag/es.jsonl (250 ítems). Cachea las
generaciones en data/benchmarks/memerag-es/llama32-3b_smoke_raw.parquet.
"""

import os
import sys
import json
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
from transformers import set_seed

from tfm_hallu import config

MEMERAG_ES = config.BENCHMARKS_DIR / "memerag-repo" / "data" / "memerag" / "es.jsonl"
OUT_DIR = config.BENCHMARKS_DIR / "memerag-es"


def rag_user_content(passages, query: str) -> str:
    """Prompt RAG fiel en español: responder SOLO con los pasajes, admitir
    explícitamente cuando no esté la respuesta. Es el prompt que vio el modelo al
    generar, y servirá como P en la futura extracción (partición P/R)."""
    blocks = "\n".join(f"{i + 1}: {p['text'].strip()}" for i, p in enumerate(passages))
    return (
        "Responde a la pregunta utilizando únicamente la información de los "
        "siguientes pasajes. Si los pasajes no contienen la respuesta, indícalo "
        "explícitamente. No añadas información que no aparezca en los pasajes.\n\n"
        f"Pasajes:\n{blocks}\n\n"
        f"Pregunta: {query.strip()}\n\nRespuesta:"
    )


def main(n: int) -> int:
    items = [json.loads(l) for l in open(MEMERAG_ES, encoding="utf-8")]
    # subconjunto reproducible (mismo criterio que el resto del pipeline: semilla 42)
    import random
    random.Random(config.SEED).shuffle(items)
    items = items[:n]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = OUT_DIR / "llama32-3b_smoke_raw.parquet"

    from tfm_hallu.extraction import load_model
    from tfm_hallu.extraction.generate import generate_from_content

    spec = config.MODELS["llama32-3b"]
    model, tok, device = load_model(spec)
    set_seed(config.SEED)

    recs, t0 = [], time.time()
    for i, it in enumerate(items):
        content = rag_user_content(it["context"], it["query"])
        n_ctx_tok = len(tok(content, add_special_tokens=False)["input_ids"])
        gen = generate_from_content(model, tok, content, device, max_new_tokens=160)
        n_tok = len(tok(gen, add_special_tokens=False)["input_ids"])
        recs.append({
            "query_id": it["query_id"], "query": it["query"],
            "context": json.dumps(it["context"], ensure_ascii=False),
            "prompt": content, "generated": gen,
            "n_ctx_tok": n_ctx_tok, "n_tok_resp": n_tok,
        })
        if (i + 1) % 10 == 0:
            print(f"  gen {i + 1}/{len(items)}  ({(time.time() - t0) / (i + 1):.2f}s/ítem)", flush=True)

    df = pd.DataFrame(recs)
    df.to_parquet(raw_path, index=False)
    dt = time.time() - t0
    print(f"\nGuardado {raw_path}  n={len(df)}  ({dt:.0f}s total, {dt / len(df):.2f}s/ítem)")
    print(f"  contexto (tokens): med={int(df.n_ctx_tok.median())}  max={int(df.n_ctx_tok.max())}")
    print(f"  respuesta (tokens): med={int(df.n_tok_resp.median())}  max={int(df.n_tok_resp.max())}")
    return 0


if __name__ == "__main__":
    nn = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    sys.exit(main(nn))
