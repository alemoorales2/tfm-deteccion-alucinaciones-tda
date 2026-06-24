"""Loader de MUCH (Dentan et al. 2025), filtrado a nuestros modelos.

MUCH publica generaciones en cuatro idiomas por cuatro modelos abiertos, con
anotaciones de factualidad a nivel de claim. Se reutilizan el texto y las
etiquetas (no los logits que también publica). Solo dos de sus modelos coinciden
con los nuestros (Llama 3.2 3B y Gemma 3 4B), los únicos de los que se puede
extraer atención; para los otros dos (Ministral-8B, Llama-3.1-8B) no hay pesos
en este trabajo.

Semántica de etiquetas (verificada en el spike 4.0): la factualidad está en
`human_labels_0/1` con valores {0 = correcto, 1 = alucinado, -1 = no verificable};
el campo `labels` por chunk (∈ {-1, 1}) es solo un marcador de claim verificable y
no se usa aquí. A nivel de muestra: alucinada (1) si algún claim tiene un human
label igual a 1; en otro caso factual (0), incluidas las muestras sin claim
verificable (que no pueden constituir alucinación bajo esta anotación).

El benchmark es de base rate baja (~4% alucinadas para nuestros modelos), sin
confound de longitud (medido en el spike). Se cachea un parquet por (modelo,
idioma) en data/benchmarks/much/ para no re-descargar; las generaciones traen
marcas de fin de turno que se limpian.
"""

import ast
import os

import pandas as pd

from .. import config
from .base import Sample

_REPO = "orailix/MUCH"
_CONFIGS_REPO = "orailix/MUCH-configs"

# model_name de MUCH -> clave de modelo de este trabajo (config.MODELS).
_MODEL_MAP = {
    "meta-llama/Llama-3.2-3B-Instruct": "llama32-3b",
    "google/gemma-3-4b-it": "gemma3-4b",
}
# Marcas de fin de turno/secuencia que algunas generaciones arrastran en `output`.
_EOS_MARKS = ("</s>", "<end_of_turn>", "<eos>", "<|eot_id|>", "<|end_of_text|>")


def _aslist(v):
    if isinstance(v, (list, tuple)):
        return list(v)
    if isinstance(v, str):
        try:
            return ast.literal_eval(v)
        except Exception:  # noqa: BLE001
            return []
    return []


def _clean(text) -> str:
    t = str(text)
    for m in _EOS_MARKS:
        t = t.replace(m, "")
    return t.strip()


def _cache_path(model_key: str, lang: str):
    return config.BENCHMARKS_DIR / "much" / f"{model_key}-{lang}.parquet"


def _build_cache() -> None:
    """Una sola pasada por MUCH (streaming) que escribe un parquet por (modelo,
    idioma) con los campos que el pipeline necesita. Filtra a nuestros dos modelos
    e idiomas en/es; ignora los logits."""
    from datasets import load_dataset

    tok = config.load_env_var("HF_TOKEN")
    if tok:
        os.environ["HF_TOKEN"] = tok

    cfgmap = {
        r["generation_config"]: r["model_name"]
        for r in load_dataset(_CONFIGS_REPO, streaming=True)["train"]
    }

    buckets: dict[tuple[str, str], list[dict]] = {}
    ds = load_dataset(_REPO, streaming=True)
    for split in ds:
        for r in ds[split]:
            mkey = _MODEL_MAP.get(cfgmap.get(r["generation_config"], ""))
            if mkey is None:
                continue
            lang = r["lang"]
            if lang not in ("en", "es"):
                continue
            hls = _aslist(r["human_labels_0"]) + _aslist(r["human_labels_1"])
            claims = [v for v in hls if v in (0, 1)]
            label = 1 if any(v == 1 for v in claims) else 0
            buckets.setdefault((mkey, lang), []).append(
                {
                    "id": str(r["generation_id"]),
                    "prompt": str(r["prompt"]),
                    "respuesta": _clean(r["output"]),
                    "label": label,
                    "wiki_url": str(r.get("wiki_url") or ""),
                    "split": split,
                    "n_claims": len(claims),
                }
            )

    out_dir = config.BENCHMARKS_DIR / "much"
    out_dir.mkdir(parents=True, exist_ok=True)
    for (mkey, lang), recs in buckets.items():
        pd.DataFrame(recs).to_parquet(out_dir / f"{mkey}-{lang}.parquet")


def load_much(model_key: str, lang: str, max_samples: int | None = None) -> list[Sample]:
    """Muestras de MUCH para un modelo y un idioma (en/es). Construye la caché en
    la primera llamada. La partición P/R se hace con plantilla de chat plana
    (P = prompt tal cual; R = generación del modelo, teacher forcing)."""
    if model_key not in _MODEL_MAP.values():
        raise ValueError(f"MUCH solo cubre {sorted(_MODEL_MAP.values())}, no {model_key!r}")
    path = _cache_path(model_key, lang)
    if not path.exists():
        _build_cache()
    df = pd.read_parquet(path)
    if max_samples is not None:
        df = df.iloc[:max_samples]

    samples = []
    for _, r in df.iterrows():
        lab = int(r["label"])
        samples.append(
            Sample(
                id=f"much-{lang}-{model_key}-{r['id']}",
                prompt=str(r["prompt"]),
                respuesta=str(r["respuesta"]),
                label=lab,
                lang=lang,
                source=f"{model_key}:{r['split']}",
                meta={"chat": "plain", "wiki_url": str(r["wiki_url"])},
            )
        )
    return samples
