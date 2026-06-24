"""20_evaluate.py MODEL BENCH METHOD - etapa SIN modelo.

Lee el intermedio persistido por 10_extract y evalúa el método con el protocolo
5-fold (Cap. 5 §5.6.1), escribiendo scores y métricas.

  toha       : selección de cabezas sin fuga + puntuación (training-free).
  lapeigvals : PCA + regresión logística ℓ2 entrenada por pliegue.

    .venv/bin/python code/scripts/20_evaluate.py phi35-mini halueval-qa toha
    .venv/bin/python code/scripts/20_evaluate.py phi35-mini halueval-qa lapeigvals
"""

import sys

import numpy as np
import pandas as pd

from tfm_hallu import config
from tfm_hallu.evaluation.metrics import detection_metrics
from tfm_hallu.evaluation.runner import metrics_by_fold, run_cv_classifier
from tfm_hallu.io import append_metrics, load_array, load_meta, load_samples, save_scores
from tfm_hallu.methods.halluzig_score import make_halluzig_estimator
from tfm_hallu.methods.lapeigvals_score import make_lapeigvals_estimator
from tfm_hallu.methods.seps_score import run_cv_seps
from tfm_hallu.methods.toha_score import run_cv_toha

# 'length' es el baseline trivial (longitud de la respuesta) para contextualizar
# el confound de longitud de HaluEval-QA (Cap. 6).
METHODS = ("toha", "lapeigvals", "halluzig", "seps", "selfcheckgpt", "length")


def main(model_key: str, bench_key: str, method: str) -> int:
    if method not in METHODS:
        raise SystemExit(f"Métodos disponibles: {METHODS}")

    samples_df = load_samples(model_key, bench_key)
    meta = load_meta(model_key, bench_key)
    y = samples_df["label"].to_numpy()
    folds = samples_df["fold"].to_numpy()
    info = None

    if method == "toha":
        headdiv = load_array(model_key, bench_key, "toha_headdiv")
        scores, fold_metrics, info = run_cv_toha(
            headdiv, y, folds,
            probe_size=config.TOHA_PROBE_SET_SIZE,
            candidates=config.TOHA_N_OPT_CANDIDATES,
            seed=config.SEED,
        )
    elif method == "lapeigvals":
        feats = load_array(model_key, bench_key, "lapeigvals_feats")
        X = feats.reshape(len(feats), -1)
        scores, fold_metrics = run_cv_classifier(
            X, y, folds,
            make_estimator=lambda n_train: make_lapeigvals_estimator(X.shape[1], n_train),
            seed=config.SEED,
        )
    elif method == "halluzig":
        X = load_array(model_key, bench_key, "halluzig_feats")
        scores, fold_metrics = run_cv_classifier(
            X, y, folds, make_estimator=make_halluzig_estimator, seed=config.SEED,
        )
    elif method == "seps":
        hidden = load_array(model_key, bench_key, "seps_hidden")
        scores, fold_metrics, info = run_cv_seps(hidden, y, folds, seed=config.SEED)
    elif method == "selfcheckgpt":
        from tfm_hallu.data import load_halueval_qa_generated
        from tfm_hallu.io.layout import result_dir
        from tfm_hallu.methods.selfcheckgpt import consistency_scores

        gens = pd.read_parquet(result_dir(model_key, bench_key) / "selfcheck_gens.parquet")
        samples_by_id = gens.groupby("sample_id")["text"].apply(list).to_dict()
        main_by_id = {s.id: s.respuesta for s in load_halueval_qa_generated(model_key)}
        # consistencia 1 - BERTScore(principal, muestra); NaN fuera del subconjunto.
        scores = consistency_scores(main_by_id, samples_by_id, samples_df["sample_id"].tolist())
        fold_metrics = metrics_by_fold(y, scores, folds)
    else:  # length (baseline trivial: longitud de la respuesta)
        scores = samples_df["n_tok_resp"].to_numpy().astype(float)
        fold_metrics = metrics_by_fold(y, scores, folds)

    save_scores(model_key, bench_key, method,
                pd.DataFrame({"sample_id": samples_df["sample_id"], "fold": folds,
                              "score": scores, "label": y}))
    append_metrics(
        [{"model": model_key, "benchmark": bench_key, "method": method, "fold": k, **m}
         for k, m in enumerate(fold_metrics)]
    )

    md = pd.DataFrame(fold_metrics)
    print(f"=== {method} · {model_key} · {bench_key}  (n={meta['n']}) ===")
    if md.empty:
        print("  (sin folds evaluables)")
    else:
        for col in ["auroc", "auprc", "f1", "fpr95"]:
            print(f"  {col:6s}: {md[col].mean():.4f} ± {md[col].std():.4f}")
    valid = ~np.isnan(np.asarray(scores, dtype=float))
    if valid.any():
        print(f"  AUROC global (n={int(valid.sum())}): "
              f"{detection_metrics(np.asarray(y)[valid], np.asarray(scores)[valid])['auroc']:.4f}")
    if method == "toha" and info is not None:
        from collections import Counter

        print(f"  N_opt por fold: {[d['n_opt'] for d in info]}")
        c = Counter(h for d in info for h in d["top_heads"])
        print(f"  cabezas (capa,cabeza) más seleccionadas: {c.most_common(5)}")
    elif method == "seps" and info is not None:
        print(f"  capa óptima por fold: {[d['layer'] for d in info]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3]))
