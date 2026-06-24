"""10_extract.py MODEL BENCH [MAX] - etapa CON modelo (Fase 1: solo TOHA).

Recorre el benchmark con el modelo cargado, extrae la atención de cada muestra,
deriva las divergencias TOHA por cabeza [n, L, H] (descartando la atención cruda
al vuelo) y persiste el intermedio compacto + el índice de muestras + meta.

    .venv/bin/python code/scripts/10_extract.py phi35-mini halueval-qa
    .venv/bin/python code/scripts/10_extract.py phi35-mini halueval-qa 100   # smoke

MAX (opcional) limita el número de muestras (para pruebas rápidas).
"""

import datetime
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import torch

from tfm_hallu import config
from tfm_hallu.data import (
    load_halueval_qa,
    load_halueval_qa_generated,
    load_memerag,
    load_much,
    load_mushroom_es,
    load_squad,
    load_xquad,
)
from tfm_hallu.evaluation.folds import assign_folds
from tfm_hallu.extraction import encode_sample, extract_attention, load_model
from tfm_hallu.io import save_barcodes, save_extraction
from tfm_hallu.methods import (
    sample_halluzig_features,
    sample_head_divergences,
    sample_lap_eigenvalues,
)
from tfm_hallu.methods.halluzig import halluzig_feature_dim


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def main(model_key: str, bench_key: str, max_override: int | None = None) -> int:
    spec = config.MODELS[model_key]
    bspec = config.BENCHMARKS[bench_key]
    max_samples = max_override if max_override is not None else bspec.max_samples

    print(f"=== Extracción TOHA+LapEigvals+HalluZig+SEPs · {model_key} · {bench_key} (max={max_samples}) ===")
    if bench_key == "halueval-qa-gen":
        samples = load_halueval_qa_generated(model_key, max_samples=max_samples)
        groups = None  # una muestra por ítem: estratificado simple
    elif bench_key == "halueval-qa":
        samples = load_halueval_qa(max_samples=max_samples)
        groups = [s.meta.get("item") for s in samples]  # par right/hallucinated -> mismo grupo
    elif bench_key in ("much-en", "much-es"):
        lang = bench_key.split("-")[1]
        samples = load_much(model_key, lang, max_samples=max_samples)
        groups = [s.meta.get("wiki_url") or s.id for s in samples]  # mismo artículo -> mismo fold
    elif bench_key == "mushroom-es":
        samples = load_mushroom_es(max_samples=max_samples)
        groups = None  # solo prueba (zero-shot): los folds no se usan al evaluar
    elif bench_key == "squad":
        samples = load_squad(model_key, max_samples=max_samples)
        groups = None  # una muestra por ítem: estratificado simple
    elif bench_key in ("xquad-en", "xquad-es"):
        lang = bench_key.split("-")[1]
        samples = load_xquad(model_key, lang, max_samples=max_samples)
        groups = None  # una muestra por ítem: estratificado simple
    elif bench_key.startswith("memerag-"):
        lang = "en" if bench_key.startswith("memerag-en") else "es"
        variant = "memerag_ext_w_majority_vote" if bench_key.endswith("majority") else "memerag"
        samples = load_memerag(lang=lang, variant=variant, max_samples=max_samples)
        groups = None  # una muestra por pregunta: estratificado simple
    else:
        raise SystemExit(f"benchmark no soportado: {bench_key}")
    labels = [s.label for s in samples]
    folds = assign_folds(labels, groups=groups)

    t_max = bspec.t_max or spec.t_max  # el benchmark puede pedir más contexto (RAG)
    print(f"Cargando {spec.hf_id} (t_max={t_max}) ...")
    model, tok, device = load_model(spec)
    # Gemma 3 es multimodal: las dimensiones del transformer de texto viven en
    # config.text_config (el config de nivel superior, Gemma3Config, no las expone).
    tcfg = getattr(model.config, "text_config", model.config)
    L = tcfg.num_hidden_layers
    H = tcfg.num_attention_heads
    n = len(samples)
    print(f"  device={device}  L={L}  H={H}  n={n}")

    k = config.LAPEIGVALS_K
    d_hz = halluzig_feature_dim()
    d_model = tcfg.hidden_size
    headdiv = np.zeros((n, L, H), dtype=np.float32)         # TOHA
    lapeig = np.zeros((n, L, H, k), dtype=np.float32)       # LapEigvals
    halluzig_feats = np.zeros((n, d_hz), dtype=np.float32)  # HalluZig
    seps_hidden = np.zeros((n, L + 1, d_model), dtype=np.float32)  # SEPs (último token/capa; fp32 por Gemma bf16)
    rows = []
    barcode_rows = []  # barcodes zigzag por muestra (re-vectorización de HalluZig sin re-extraer)
    t_total = 0.0
    for i, s in enumerate(samples):
        enc = encode_sample(tok, s, t_max)
        ok = True
        t0 = time.time()
        try:
            attn, hidden = extract_attention(model, enc, device)
            headdiv[i] = sample_head_divergences(attn, enc.n_prompt)
            lapeig[i] = sample_lap_eigenvalues(attn, k)
            hz_feats, hz_bc, hz_levels = sample_halluzig_features(attn, return_barcode=True)
            halluzig_feats[i] = hz_feats
            barcode_rows.append({
                "sample_id": s.id, "n_levels": int(hz_levels),
                "h0": json.dumps(hz_bc[0].tolist()), "h1": json.dumps(hz_bc[1].tolist()),
            })
            seps_hidden[i] = hidden
            del attn
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  [WARN] muestra {s.id}: {type(e).__name__}: {e}")
        t_total += time.time() - t0
        rows.append(
            {
                "sample_id": s.id,
                "label": s.label,
                "lang": s.lang,
                "source": s.source,
                "n_tok_prompt": enc.n_prompt,
                "n_tok_resp": enc.n_tok_resp,
                "fold": int(folds[i]),
                "truncated": enc.truncated,
                "ok": ok,
            }
        )
        if device == "mps":
            torch.mps.empty_cache()
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n}  {t_total / (i + 1):.2f}s/muestra")

    samples_df = pd.DataFrame(rows)
    meta = {
        "model_key": model_key,
        "model_id": spec.hf_id,
        "benchmark": bench_key,
        "seed": config.SEED,
        "t_max": t_max,
        "n": n,
        "L": L,
        "H": H,
        "n_folds": config.N_FOLDS,
        "commit": _git_commit(),
        "date": datetime.date.today().isoformat(),
        "seconds_per_sample": round(t_total / max(n, 1), 4),
        "n_truncated": int(samples_df["truncated"].sum()),
        "n_failed": int((~samples_df["ok"]).sum()),
        "lapeigvals_k": k,
        "halluzig_percentile": config.HALLUZIG_PERCENTILE,
        "halluzig_pi_res": config.HALLUZIG_PI_RES,
        "halluzig_pi_sigma": config.HALLUZIG_PI_SIGMA,
        "halluzig_betti_points": config.HALLUZIG_BETTI_POINTS,
        "halluzig_dim": d_hz,
        "d_model": d_model,
        "n_hidden_states": L + 1,
    }
    arrays = {
        "toha_headdiv": headdiv,
        "lapeigvals_feats": lapeig,
        "halluzig_feats": halluzig_feats,
        "seps_hidden": seps_hidden,
    }
    save_extraction(model_key, bench_key, arrays, samples_df, meta)
    if barcode_rows:
        save_barcodes(model_key, bench_key, barcode_rows)
    print(
        f"\nGuardado en data/results/{model_key}/{bench_key}/  "
        f"({meta['seconds_per_sample']}s/muestra, "
        f"truncadas={meta['n_truncated']}, fallidas={meta['n_failed']})"
    )
    return 0


if __name__ == "__main__":
    mk = sys.argv[1]
    bk = sys.argv[2]
    mx = int(sys.argv[3]) if len(sys.argv) > 3 else None
    sys.exit(main(mk, bk, mx))
