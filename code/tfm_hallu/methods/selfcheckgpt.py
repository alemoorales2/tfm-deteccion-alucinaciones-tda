"""SelfCheckGPT: consistencia por BERTScore (etapa SIN modelo).

Manakul et al. (2023): se muestrean N respuestas adicionales por pregunta y se mide
la consistencia entre la respuesta principal y las muestras. Una respuesta factual
es estable (el modelo repite la misma información), mientras que una alucinación
varía entre muestras. La consistencia se mide con BERTScore (la variante que los
autores reportan como más equilibrada). El score de alucinación es
1 - (media de BERTScore-F1 entre la respuesta principal y cada muestra): alto cuando
las muestras discrepan de la principal.

El modelo de BERTScore es multilingüe (config.BERTSCORE_MODEL) para reutilizarse en
los benchmarks en español (MUCH, Mu-SHROOM). Se usa F1 sin rescale de baseline: para
el AUROC solo importa el orden de los scores.
"""

import numpy as np

from .. import config


def consistency_scores(main_by_id: dict, samples_by_id: dict, ids: list) -> np.ndarray:
    """Score de alucinación por muestra, alineado a `ids`.

    `main_by_id`: id -> respuesta principal. `samples_by_id`: id -> lista de muestras.
    Devuelve un array de longitud len(ids) con 1 - media de BERTScore-F1; NaN para los
    ids sin muestras. Una sola llamada a bert_score sobre todos los pares (principal,
    muestra) para amortizar la carga del modelo.
    """
    from bert_score import score as bertscore

    cands, refs, owner = [], [], []
    for i, sid in enumerate(ids):
        if sid not in samples_by_id:
            continue
        main = (main_by_id.get(sid) or "").strip() or "[empty]"
        for s in samples_by_id[sid]:
            s = (str(s) or "").strip() or "[empty]"
            cands.append(main)
            refs.append(s)
            owner.append(i)

    scores = np.full(len(ids), np.nan, dtype=np.float64)
    if not cands:
        return scores

    _, _, f1 = bertscore(
        cands, refs,
        model_type=config.BERTSCORE_MODEL,
        num_layers=config.BERTSCORE_NUM_LAYERS,
        verbose=False,
    )
    f1 = f1.numpy()
    owner = np.array(owner)
    for i in range(len(ids)):
        fi = f1[owner == i]
        if fi.size:
            scores[i] = 1.0 - float(fi.mean())
    return scores
