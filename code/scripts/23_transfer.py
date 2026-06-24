"""23_transfer.py MODEL TRAIN_BENCH TEST_BENCH - transferencia cross-language/cross-domain.

Generaliza `22_zeroshot.py` (que estaba cableado a MUCH-ES→Mu-SHROOM-ES): entrena
cada método sobre TRAIN_BENCH y lo aplica sin reajuste a TEST_BENCH, del MISMO modelo
(las dimensiones de los intermedios coinciden por compartir L/H/d_model). Pensado para
la transferibilidad cross-language EN→ES en el régimen con contexto (Cap. 6, Fase 5):

    .venv/bin/python code/scripts/23_transfer.py llama32-3b xquad-en xquad-es
    .venv/bin/python code/scripts/23_transfer.py llama32-3b xquad-es xquad-en

Guarda scores y métricas bajo el benchmark `<TEST>-from-<TRAIN-lang>` para trazabilidad
(p. ej. `xquad-es-from-en`), sin pisar la CV interna.
"""

import sys

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

METHODS = ("toha", "lapeigvals", "halluzig", "seps")


def _transfer_one(model_key, train_bench, test_bench, method, y_tr, y_te):
    if method == "toha":
        a_tr = load_array(model_key, train_bench, "toha_headdiv")
        a_te = load_array(model_key, test_bench, "toha_headdiv")
        return transfer_toha(a_tr, y_tr, a_te, config.TOHA_PROBE_SET_SIZE,
                             config.TOHA_N_OPT_CANDIDATES, config.SEED)
    if method == "lapeigvals":
        x_tr = load_array(model_key, train_bench, "lapeigvals_feats").reshape(len(y_tr), -1)
        x_te = load_array(model_key, test_bench, "lapeigvals_feats").reshape(len(y_te), -1)
        return transfer_classifier(
            x_tr, y_tr, x_te,
            make_estimator=lambda n: make_lapeigvals_estimator(x_tr.shape[1], n), seed=config.SEED,
        )
    if method == "halluzig":
        x_tr = load_array(model_key, train_bench, "halluzig_feats")
        x_te = load_array(model_key, test_bench, "halluzig_feats")
        return transfer_classifier(x_tr, y_tr, x_te, make_estimator=make_halluzig_estimator,
                                   seed=config.SEED)
    h_tr = load_array(model_key, train_bench, "seps_hidden")
    h_te = load_array(model_key, test_bench, "seps_hidden")
    return transfer_seps(h_tr, y_tr, h_te, config.SEED, config.SEPS_VAL_FRAC)


def main(model_key: str, train_bench: str, test_bench: str) -> int:
    s_tr = load_samples(model_key, train_bench)
    s_te = load_samples(model_key, test_bench)
    y_tr = s_tr["label"].to_numpy()
    y_te = s_te["label"].to_numpy()
    train_lang = config.BENCHMARKS[train_bench].lang
    out_bench = f"{test_bench}-from-{train_lang}"

    print(f"=== Transferencia {train_bench}→{test_bench} · {model_key}  "
          f"(train n={len(y_tr)} pos={int(y_tr.sum())} | test n={len(y_te)} pos={int(y_te.sum())}) ===")
    rows = []
    for method in METHODS:
        scores, info = _transfer_one(model_key, train_bench, test_bench, method, y_tr, y_te)
        m = detection_metrics(y_te, scores)
        m["n_eval"] = int(len(y_te))
        save_scores(model_key, out_bench, method,
                    pd.DataFrame({"sample_id": s_te["sample_id"], "fold": -1, "score": scores, "label": y_te}))
        append_metrics([{"model": model_key, "benchmark": out_bench, "method": method, "fold": -1, **m}])
        rows.append((method, m["auroc"], m["auprc"], m["f1"]))
        print(f"  {method:11s} auroc={m['auroc']:.4f}  auprc={m['auprc']:.4f}  f1={m['f1']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
