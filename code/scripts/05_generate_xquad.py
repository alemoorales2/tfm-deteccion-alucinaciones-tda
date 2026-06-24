"""05_generate_xquad.py MODEL LANG [MAX] - genera respuestas open-book sobre XQuAD y las etiqueta.

XQuAD (Artetxe et al. 2020) es comprensión lectora paralela en once idiomas (mismas
preguntas y contextos traducidos profesionalmente). Aquí se usa la pareja EN/ES para
la comparación cross-language en el régimen CON contexto (donde rinden los topológicos),
con preguntas idénticas salvo el idioma. Mismo protocolo que `05_generate_squad.py`: el
modelo responde con el contexto en el prompt y se etiqueta híbrido (exact-match/F1 estilo
SQuAD, adaptado al español en `xquad_is_correct`, + juez Gemini para las ambiguas).

    .venv/bin/python code/scripts/05_generate_xquad.py llama32-3b es      # 1190 ítems ES
    .venv/bin/python code/scripts/05_generate_xquad.py llama32-3b en 10   # smoke

Cachea las generaciones (<modelo>_raw.parquet) ANTES de etiquetar y los veredictos del
juez (<modelo>_verdicts.parquet): un corte por cuota no pierde trabajo. Resultado en
data/benchmarks/xquad-<lang>/<modelo>.parquet.
"""

import os
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
from transformers import set_seed

from tfm_hallu import config
from tfm_hallu.data.xquad import xquad_is_correct, xquad_user_content
from tfm_hallu.labeling import judge_batch


def _load_xquad_items(lang: str, n: int | None):
    from datasets import load_dataset

    tok = config.load_env_var("HF_TOKEN")
    if tok:
        os.environ["HF_TOKEN"] = tok
    ds = load_dataset("google/xquad", f"xquad.{lang}", split="validation")
    # n=None -> dataset entero en orden (garantiza el mismo subconjunto/orden que el
    # otro idioma, que es paralelo por id). Con n se muestrea reproducible.
    if n is not None and n < len(ds):
        ds = ds.shuffle(seed=config.SEED).select(range(n))
    return ds


def _generate(model_key: str, lang: str, items, raw_path):
    from tfm_hallu.extraction import load_model
    from tfm_hallu.extraction.generate import generate_from_content

    spec = config.MODELS[model_key]
    model, tok, device = load_model(spec)
    set_seed(config.SEED)
    recs, t0 = [], time.time()
    for i, it in enumerate(items):
        golds = list(dict.fromkeys(it["answers"]["text"]))
        content = xquad_user_content(it["context"], it["question"], lang)
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


def _hybrid_label(recs, lang: str, verdict_path):
    """exact-match/F1 -> factual (0) en los casos claros; el resto lo juzga Gemini en
    lotes (reanudable, veredictos cacheados por id). Devuelve (labels, stats)."""
    labels = [None] * len(recs)
    ambiguous = []
    for i, r in enumerate(recs):
        if xquad_is_correct(r["gold"].split(" | "), r["generated"], lang):
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
            # ids locales enteros para el juez; se mapean de vuelta al id de XQuAD
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


def main(model_key: str, lang: str, max_items: int | None = None) -> int:
    if lang not in ("en", "es"):
        raise SystemExit(f"lang debe ser 'en' o 'es', no {lang!r}")
    out_dir = config.BENCHMARKS_DIR / f"xquad-{lang}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_key}.parquet"
    raw_path = out_dir / f"{model_key}_raw.parquet"
    verdict_path = out_dir / f"{model_key}_verdicts.parquet"

    if raw_path.exists():
        recs = pd.read_parquet(raw_path).to_dict("records")
        print(f"=== {model_key} xquad-{lang}: reusando {len(recs)} generaciones cacheadas ===")
    else:
        items = _load_xquad_items(lang, max_items)
        print(f"=== Generación open-book XQuAD-{lang} · {model_key} · {len(items)} ítems ===")
        recs = _generate(model_key, lang, items, raw_path)

    print("Etiquetando (exact-match/F1 + juez Gemini para las ambiguas, reanudable)...")
    labels, stats = _hybrid_label(recs, lang, verdict_path)
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
    lg = sys.argv[2]
    mx = int(sys.argv[3]) if len(sys.argv) > 3 else None
    sys.exit(main(mk, lg, mx))
