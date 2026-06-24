"""05_generate_squad.py MODEL [MAX] - genera respuestas open-book sobre SQuAD y las etiqueta.

Control de reproducción de TOHA (Fase 4.9). Para cada ítem de SQuAD (contexto +
pregunta), el modelo genera una respuesta CON el contexto en el prompt. El etiquetado
es híbrido (como en HaluEval generado): exact-match normalizado / F1 estilo SQuAD para
los factuales claros, y juez Gemini solo para las aparentes alucinaciones (resuelve
número↔palabra «10»/«ten», morfología «Jews»/«Jewish» y paráfrasis, que las reglas no
capturan y que, sin juez, inflan la clase alucinada con falsos positivos).

Cachea las generaciones (<modelo>_raw.parquet) ANTES de etiquetar y los veredictos del
juez (<modelo>_verdicts.parquet), de modo que un corte por cuota no pierde trabajo y el
reetiquetado no exige regenerar. Resultado en data/benchmarks/squad/<modelo>.parquet.

    .venv/bin/python code/scripts/05_generate_squad.py llama32-3b        # 800 ítems
    .venv/bin/python code/scripts/05_generate_squad.py llama32-3b 10     # smoke
"""

import os
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
from transformers import set_seed

from tfm_hallu import config
from tfm_hallu.data.squad import squad_is_correct, squad_user_content
from tfm_hallu.labeling import judge_batch

DEFAULT_N = 800


def _load_squad_items(n: int):
    from datasets import load_dataset

    tok = config.load_env_var("HF_TOKEN")
    if tok:
        os.environ["HF_TOKEN"] = tok
    ds = load_dataset("rajpurkar/squad", split="validation")
    ds = ds.shuffle(seed=config.SEED).select(range(min(n, len(ds))))
    return ds


def _generate(model_key: str, items, raw_path):
    from tfm_hallu.extraction import load_model
    from tfm_hallu.extraction.generate import generate_from_content

    spec = config.MODELS[model_key]
    model, tok, device = load_model(spec)
    set_seed(config.SEED)
    recs, t0 = [], time.time()
    for i, it in enumerate(items):
        golds = list(dict.fromkeys(it["answers"]["text"]))
        content = squad_user_content(it["context"], it["question"])
        gen = generate_from_content(model, tok, content, device)
        n_tok = len(tok(gen, add_special_tokens=False)["input_ids"])
        recs.append({"id": it["id"], "question": it["question"], "gold": " | ".join(golds),
                     "prompt": content, "generated": gen, "n_tok_resp": n_tok})
        if (i + 1) % 50 == 0:
            print(f"  gen {i + 1}/{len(items)}  ({(time.time() - t0) / (i + 1):.2f}s/ítem)")
    pd.DataFrame(recs).to_parquet(raw_path, index=False)  # cachear ANTES de etiquetar
    return recs


def _load_verdicts(path) -> dict:
    if path.exists():
        vc = pd.read_parquet(path)
        return {str(k): bool(v) for k, v in zip(vc["id"], vc["is_hall"])}
    return {}


def _save_verdicts(path, cache: dict):
    pd.DataFrame({"id": list(cache), "is_hall": list(cache.values())}).to_parquet(path, index=False)


def _hybrid_label(recs, verdict_path):
    """exact-match/F1 -> factual (0) en los casos claros; el resto lo juzga Gemini en
    lotes (reanudable, veredictos cacheados por id). Devuelve (labels, stats)."""
    labels = [None] * len(recs)
    ambiguous = []
    for i, r in enumerate(recs):
        if squad_is_correct(r["gold"].split(" | "), r["generated"]):
            labels[i] = 0
        else:
            ambiguous.append(i)

    cache = _load_verdicts(verdict_path)
    pending = [i for i in ambiguous if recs[i]["id"] not in cache]
    for i in ambiguous:
        if recs[i]["id"] in cache:
            labels[i] = 1 if cache[recs[i]["id"]] else 0

    judged_now = 0
    if pending:
        from tfm_hallu.labeling import _gemini_client

        client = _gemini_client()
        bs = config.JUDGE_BATCH_SIZE
        for b in range(0, len(pending), bs):
            chunk = pending[b : b + bs]
            # ids locales enteros para el juez; se mapean de vuelta al id de SQuAD
            items = [(k, recs[i]["question"], recs[i]["gold"], recs[i]["generated"]) for k, i in enumerate(chunk)]
            try:
                verdicts = judge_batch(client, items)
            except Exception as e:  # noqa: BLE001
                print(f"  [STOP] juez interrumpido ({type(e).__name__}): {str(e)[:140]}")
                break
            for k, i in enumerate(chunk):
                ish = bool(verdicts.get(k, True))
                cache[recs[i]["id"]] = ish
                labels[i] = 1 if ish else 0
            judged_now += len(chunk)
            _save_verdicts(verdict_path, cache)
            if b + bs < len(pending):
                time.sleep(config.JUDGE_SLEEP_SECONDS)

    remaining = sum(1 for v in labels if v is None)
    return labels, {"ambiguous": len(ambiguous), "judged_now": judged_now, "remaining": remaining}


def main(model_key: str, max_items: int = DEFAULT_N) -> int:
    out_dir = config.BENCHMARKS_DIR / "squad"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_key}.parquet"
    raw_path = out_dir / f"{model_key}_raw.parquet"
    verdict_path = out_dir / f"{model_key}_verdicts.parquet"

    if raw_path.exists():
        recs = pd.read_parquet(raw_path).to_dict("records")
        print(f"=== {model_key}: reusando {len(recs)} generaciones cacheadas ===")
    else:
        items = _load_squad_items(max_items)
        print(f"=== Generación open-book SQuAD · {model_key} · {len(items)} ítems ===")
        recs = _generate(model_key, items, raw_path)

    print("Etiquetando (exact-match/F1 + juez Gemini para las ambiguas, reanudable)...")
    labels, stats = _hybrid_label(recs, verdict_path)
    print(f"  ambiguas={stats['ambiguous']}  juzgadas ahora={stats['judged_now']}  pendientes={stats['remaining']}")
    if stats["remaining"] > 0:
        print("\n⚠ Etiquetado PARCIAL (cuota del juez). Re-ejecuta para continuar con lo pendiente.")
        return 1

    for r, lab in zip(recs, labels):
        r["label"] = lab
    df = pd.DataFrame(recs)
    df.to_parquet(out_path, index=False)
    bal = df["label"].value_counts().to_dict()
    print(f"\nGuardado {out_path}  n={len(df)}  balance {{0:factual,1:alucinada}}={bal}")
    c, h = df[df.label == 0]["n_tok_resp"], df[df.label == 1]["n_tok_resp"]
    if len(c) and len(h):
        print(f"  long. respuesta (tokens)  factual: med={c.median():.0f}  alucinada: med={h.median():.0f}")
    return 0


if __name__ == "__main__":
    mk = sys.argv[1]
    mx = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_N
    sys.exit(main(mk, mx))
