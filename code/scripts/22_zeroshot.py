"""22_zeroshot.py MODEL METHOD - evaluación zero-shot (etapa SIN modelo).

Entrena el método sobre MUCH-ES y lo aplica, sin reajuste, a Mu-SHROOM-ES (Cap. 5
§5.6: transferibilidad dentro del español, entre dominios de pregunta). Train y
test son del mismo modelo. Guarda scores y métricas bajo el benchmark
`mushroom-es-zeroshot` (separado del `mushroom-es` de la CV interna).

    .venv/bin/python code/scripts/22_zeroshot.py llama32-3b toha
"""

import sys

import numpy as np
import pandas as pd

from tfm_hallu import config
from tfm_hallu.evaluation.metrics import detection_metrics
from tfm_hallu.evaluation.transfer import (
    transfer_classifier,
    transfer_seps,
    transfer_toha,
)
from tfm_hallu.io import append_metrics, load_array, load_samples, save_scores
from tfm_hallu.methods.halluzig_score import make_halluzig_estimator
from tfm_hallu.methods.lapeigvals_score import make_lapeigvals_estimator

TRAIN_BENCH = "much-es"
TEST_BENCH = "mushroom-es"
OUT_BENCH = "mushroom-es-zeroshot"
METHODS = ("toha", "lapeigvals", "halluzig", "seps")


def main(model_key: str, method: str) -> int:
    if method not in METHODS:
        raise SystemExit(f"Métodos zero-shot: {METHODS}")

    s_tr = load_samples(model_key, TRAIN_BENCH)
    s_te = load_samples(model_key, TEST_BENCH)
    y_tr = s_tr["label"].to_numpy()
    y_te = s_te["label"].to_numpy()
    info = {}

    if method == "toha":
        a_tr = load_array(model_key, TRAIN_BENCH, "toha_headdiv")
        a_te = load_array(model_key, TEST_BENCH, "toha_headdiv")
        scores, info = transfer_toha(
            a_tr, y_tr, a_te, config.TOHA_PROBE_SET_SIZE, config.TOHA_N_OPT_CANDIDATES, config.SEED
        )
    elif method == "lapeigvals":
        x_tr = load_array(model_key, TRAIN_BENCH, "lapeigvals_feats").reshape(len(s_tr), -1)
        x_te = load_array(model_key, TEST_BENCH, "lapeigvals_feats").reshape(len(s_te), -1)
        scores, info = transfer_classifier(
            x_tr, y_tr, x_te,
            make_estimator=lambda n: make_lapeigvals_estimator(x_tr.shape[1], n),
            seed=config.SEED,
        )
    elif method == "halluzig":
        x_tr = load_array(model_key, TRAIN_BENCH, "halluzig_feats")
        x_te = load_array(model_key, TEST_BENCH, "halluzig_feats")
        scores, info = transfer_classifier(
            x_tr, y_tr, x_te, make_estimator=make_halluzig_estimator, seed=config.SEED
        )
    else:  # seps
        h_tr = load_array(model_key, TRAIN_BENCH, "seps_hidden")
        h_te = load_array(model_key, TEST_BENCH, "seps_hidden")
        scores, info = transfer_seps(h_tr, y_tr, h_te, config.SEED, config.SEPS_VAL_FRAC)

    m = detection_metrics(y_te, scores)
    m["n_eval"] = int(len(y_te))
    save_scores(
        model_key, OUT_BENCH, method,
        pd.DataFrame({"sample_id": s_te["sample_id"], "fold": -1, "score": scores, "label": y_te}),
    )
    append_metrics([{"model": model_key, "benchmark": OUT_BENCH, "method": method, "fold": -1, **m}])
    print(f"=== {method} zero-shot · {model_key} · {TRAIN_BENCH}→{TEST_BENCH}  (n={len(y_te)}, pos={int(y_te.sum())}) ===")
    for c in ("auroc", "auprc", "f1", "fpr95"):
        print(f"  {c:6s}: {m[c]:.4f}")
    if info:
        print(f"  info: {info if method != 'toha' else {'n_opt': info['n_opt']}}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
