"""05_generate.py MODEL [MAX] - genera respuestas closed-book y las etiqueta.

Construye el «HaluEval-QA generado» de un modelo: por cada pregunta (closed-book)
el modelo genera una respuesta y se etiqueta como factual/alucinada con el híbrido
exact-match + juez Gemini. Se cachea en data/benchmarks/halueval-qa-gen/<modelo>.parquet
(gitignored). A diferencia de las respuestas dadas, aquí hay UNA muestra por ítem.

    .venv/bin/python code/scripts/05_generate.py phi35-mini          # 1000 ítems
    .venv/bin/python code/scripts/05_generate.py phi35-mini 100      # prueba

Requiere GEMINI_KEY en .env (juez). Cuidado con el rate limit del plan gratuito:
el juez se llama en lotes espaciados.
"""

import json
import os
import random
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
from transformers import set_seed

from tfm_hallu import config
from tfm_hallu.extraction import load_model
from tfm_hallu.extraction.generate import generate_answer
from tfm_hallu.labeling import hybrid_label


def main(model_key: str, max_items: int = 1000) -> int:
    spec = config.MODELS[model_key]
    src = config.BENCHMARKS_DIR / "halueval-qa" / "qa_data.json"
    out_dir = config.BENCHMARKS_DIR / "halueval-qa-gen"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_key}.parquet"

    raw_path = out_dir / f"{model_key}_raw.parquet"
    verdict_path = out_dir / f"{model_key}_verdicts.parquet"

    # --- Generación (cacheada): si ya está, se reutiliza y no se carga el modelo ---
    if raw_path.exists():
        recs = pd.read_parquet(raw_path).to_dict("records")
        print(f"=== {model_key}: reusando {len(recs)} generaciones cacheadas ({raw_path.name}) ===")
    else:
        items_all = [json.loads(line) for line in src.open(encoding="utf-8") if line.strip()]
        items = random.Random(config.SEED).sample(items_all, min(max_items, len(items_all)))
        print(f"=== Generación closed-book · {model_key} · {len(items)} ítems ===")
        model, tok, device = load_model(spec)
        set_seed(config.SEED)
        recs = []
        t0 = time.time()
        for i, it in enumerate(items):
            gen = generate_answer(model, tok, it["question"], device)
            n_tok = len(tok(gen, add_special_tokens=False)["input_ids"])
            recs.append({"item": i, "question": it["question"], "right_answer": it["right_answer"],
                         "generated": gen, "n_tok_resp": n_tok})
            if (i + 1) % 50 == 0:
                print(f"  gen {i + 1}/{len(items)}  ({(time.time() - t0) / (i + 1):.2f}s/ítem)")
        pd.DataFrame(recs).to_parquet(raw_path, index=False)  # cachear ANTES de etiquetar
        print(f"  generaciones cacheadas en {raw_path.name}")

    # --- Etiquetado híbrido reanudable (exact-match + juez Gemini) ---
    print("Etiquetando (exact-match + juez Gemini en lotes, reanudable)...")
    labels, stats = hybrid_label(recs, verdict_cache_path=verdict_path)
    print(f"  ambiguas={stats['ambiguous']}  juzgadas ahora={stats['judged_now']}  pendientes={stats['remaining']}")

    if stats["remaining"] > 0:
        print(f"\n⚠ Etiquetado PARCIAL (probable cuota del juez). Generaciones y veredictos cacheados; "
              f"re-ejecuta el mismo comando para continuar solo con lo pendiente.")
        return 1

    for r, lab in zip(recs, labels):
        r["label"] = lab
    df = pd.DataFrame(recs)
    df.to_parquet(out_path, index=False)

    bal = df["label"].value_counts().to_dict()
    print(f"\nGuardado {out_path}  n={len(df)}  balance {{0:factual,1:alucinada}}={bal}")
    c, h = df[df.label == 0]["n_tok_resp"], df[df.label == 1]["n_tok_resp"]
    print(f"  long. respuesta (tokens)  factual: med={c.median():.0f} media={c.mean():.1f}  "
          f"alucinada: med={h.median():.0f} media={h.mean():.1f}")
    return 0


if __name__ == "__main__":
    mk = sys.argv[1]
    mx = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    sys.exit(main(mk, mx))
