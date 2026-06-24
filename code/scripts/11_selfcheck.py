"""11_selfcheck.py MODEL [N_SUBSET] - etapa CON modelo: muestreo de SelfCheckGPT.

Genera SELFCHECK_N respuestas adicionales (closed-book, temperatura
SELFCHECK_TEMPERATURE) para un subconjunto estratificado del benchmark
halueval-qa-gen y las persiste como texto en `selfcheck_gens.parquet`. La
consistencia por BERTScore se calcula después, sin modelo, en 20_evaluate.

Reanudable: los sample_id que ya tengan sus N muestras no se regeneran.

    .venv/bin/python code/scripts/11_selfcheck.py phi35-mini
    .venv/bin/python code/scripts/11_selfcheck.py phi35-mini 12   # smoke
"""

import os
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd

from tfm_hallu import config
from tfm_hallu.data import load_halueval_qa_generated
from tfm_hallu.extraction import load_model
from tfm_hallu.extraction.generate import generate_answer
from tfm_hallu.io.layout import result_dir

BENCH = "halueval-qa-gen"


def _stratified_subset(samples, k, seed=config.SEED):
    rng = np.random.RandomState(seed)
    labels = np.array([s.label for s in samples])
    pos = rng.permutation(np.where(labels == 1)[0])
    neg = rng.permutation(np.where(labels == 0)[0])
    kp = round(k * labels.mean())
    sel = np.concatenate([pos[:kp], neg[: k - kp]])
    sel.sort()
    return [samples[i] for i in sel]


def main(model_key: str, n_subset: int | None = None) -> int:
    n_subset = config.SELFCHECK_SUBSET if n_subset is None else n_subset
    N = config.SELFCHECK_N
    samples = load_halueval_qa_generated(model_key)
    subset = _stratified_subset(samples, min(n_subset, len(samples)))

    out_path = result_dir(model_key, BENCH, create=True) / "selfcheck_gens.parquet"
    rows, done = [], {}
    if out_path.exists():
        prev = pd.read_parquet(out_path)
        rows = prev.to_dict("records")
        done = prev.groupby("sample_id").size().to_dict()
    todo = [s for s in subset if done.get(s.id, 0) < N]

    print(f"=== SelfCheckGPT · {model_key} · {BENCH} ===")
    print(f"  subconjunto={len(subset)}  N={N}  T={config.SELFCHECK_TEMPERATURE}  "
          f"ya hechas={len(subset) - len(todo)}  por generar={len(todo)}")
    if not todo:
        print("  nada que generar."); return 0

    spec = config.MODELS[model_key]
    print(f"Cargando {spec.hf_id} ...")
    model, tok, device = load_model(spec)

    t0, ngen = time.time(), 0
    for c, s in enumerate(todo):
        for j in range(done.get(s.id, 0), N):
            txt = generate_answer(
                model, tok, s.prompt, device,
                max_new_tokens=config.QA_GEN_MAX_NEW_TOKENS,
                temperature=config.SELFCHECK_TEMPERATURE,
            )
            rows.append({"sample_id": s.id, "gen_idx": j, "text": txt})
            ngen += 1
        if (c + 1) % 10 == 0 or c + 1 == len(todo):
            pd.DataFrame(rows).to_parquet(out_path, index=False)
            print(f"  {c + 1}/{len(todo)}  ({ngen} gens, {(time.time() - t0) / max(ngen, 1):.2f}s/gen)")

    el = time.time() - t0
    print(f"\nGuardado {out_path}  ({ngen} generaciones nuevas, {el:.0f}s, "
          f"{el / max(ngen, 1):.2f}s/gen)")
    return 0


if __name__ == "__main__":
    mk = sys.argv[1]
    ns = int(sys.argv[2]) if len(sys.argv) > 2 else None
    sys.exit(main(mk, ns))
