"""Evaluación zero-shot: entrenar en un benchmark, aplicar a otro (Fase 4).

Variante «train-en-A / aplica-a-B» de los métodos entrenables, reutilizando la
lógica de selección sin fuga de los módulos de score (la selección de cabezas de
TOHA, de capa de SEPs y el ajuste del clasificador se hacen solo sobre el train).
Se usa para Mu-SHROOM-ES (Cap. 5 §5.6): clasificadores entrenados en MUCH-ES,
aplicados sin reajuste. Train y test son del MISMO modelo, de modo que las
dimensiones (L·H·k de LapEigvals, L×d de SEPs, D fija de HalluZig, L·H de TOHA)
coinciden por construcción.
"""

import numpy as np

from ..methods.seps_score import _make_probe, _select_layer
from ..methods.toha_score import _select_n_opt, _stratified_probe


def transfer_toha(headdiv_tr, y_tr, headdiv_te, probe_size, candidates, seed):
    """Selecciona hallucination-aware heads + N_opt sobre el train (probe set +
    AUROC en train) y puntúa el test promediando sobre esas cabezas."""
    y_tr = np.asarray(y_tr)
    n_tr, L, H = headdiv_tr.shape
    div_tr = headdiv_tr.reshape(n_tr, L * H)
    div_te = headdiv_te.reshape(len(headdiv_te), L * H)
    train_idx = np.arange(n_tr)
    probe_idx = _stratified_probe(y_tr, train_idx, probe_size, seed)
    p = div_tr[probe_idx]
    deltas = p[y_tr[probe_idx] == 1].mean(axis=0) - p[y_tr[probe_idx] == 0].mean(axis=0)
    ranked = np.argsort(-deltas)
    n_opt, _ = _select_n_opt(div_tr, y_tr, train_idx, ranked, candidates)
    heads = ranked[:n_opt]
    return div_te[:, heads].mean(axis=1), {"n_opt": int(n_opt),
                                           "heads": [(int(h // H), int(h % H)) for h in heads]}


def transfer_classifier(X_tr, y_tr, X_te, make_estimator, seed=42):
    """Ajusta un clasificador sobre todo el train y devuelve P(alucinación) del test."""
    est = make_estimator(len(X_tr))
    est.fit(np.asarray(X_tr), np.asarray(y_tr))
    return est.predict_proba(np.asarray(X_te))[:, 1], {}


def transfer_seps(hidden_tr, y_tr, hidden_te, seed, val_frac):
    """Elige la capa por holdout interno del train, reentrena la sonda en todo el
    train y puntúa el test con esa capa."""
    hidden_tr = np.asarray(hidden_tr, dtype=np.float32)
    hidden_te = np.asarray(hidden_te, dtype=np.float32)
    y_tr = np.asarray(y_tr)
    l_opt, auc_val = _select_layer(hidden_tr, y_tr, np.arange(len(hidden_tr)), seed, val_frac)
    probe = _make_probe(seed)
    probe.fit(hidden_tr[:, l_opt, :], y_tr)
    s = probe.predict_proba(hidden_te[:, l_opt, :])[:, 1]
    return s, {"layer": int(l_opt), "auc_val": float(auc_val)}
