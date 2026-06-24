"""Configuración central del motor: semilla, rutas y registros.

Reúne en un solo sitio las constantes del Capítulo 5 (§5.2, §5.6) y del diseño
(docs/diseno_pipeline_cap6.md §7) para que el resto del paquete no las repita.
"""

import os
from dataclasses import dataclass
from pathlib import Path

# Semilla única en todas las fuentes de aleatoriedad (Cap. 5 §5.6.3).
SEED = 42

# --- Rutas -----------------------------------------------------------------
# config.py vive en code/tfm_hallu/, así que la raíz del repo es parents[2].
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
BENCHMARKS_DIR = DATA_DIR / "benchmarks"
RESULTS_DIR = DATA_DIR / "results"

# --- Inferencia (Cap. 5 §5.2 / diseño §7) ----------------------------------
GEN_TEMPERATURE = 0.7
GEN_TOP_P = 0.9
GEN_MAX_NEW_TOKENS = 200
ATTN_IMPLEMENTATION = "eager"  # obligatorio para output_attentions


@dataclass(frozen=True)
class ModelSpec:
    """Un modelo del barrido. `loader` distingue la carga (Gemma 3 va en modo
    texto con su clase multimodal; el resto, AutoModelForCausalLM)."""

    key: str
    hf_id: str
    dtype: str  # "float16" | "bfloat16"
    t_max: int  # longitud máxima de secuencia (truncado de la atención)
    loader: str = "causal"  # "causal" | "gemma3_text"


# T_max y dtype por modelo: diseño §7 (Gemma en bfloat16, resto FP16).
MODELS = {
    "phi35-mini": ModelSpec("phi35-mini", "microsoft/Phi-3.5-mini-instruct", "float16", 384),
    "llama32-3b": ModelSpec("llama32-3b", "meta-llama/Llama-3.2-3B-Instruct", "float16", 512),
    "gemma3-4b": ModelSpec("gemma3-4b", "google/gemma-3-4b-it", "bfloat16", 512, "gemma3_text"),
}


@dataclass(frozen=True)
class BenchmarkSpec:
    """Un benchmark. `max_samples` es el tope de muestras (no de ítems); None
    significa usarlo entero (MUCH-ES, Mu-SHROOM-ES)."""

    key: str
    lang: str
    max_samples: int | None
    t_max: int | None = None  # override del t_max del modelo (RAG: contexto largo)


# HaluEval-QA con respuestas dadas (confundido por longitud, solo diagnóstico),
# HaluEval-QA generado por el modelo (respuestas propias + etiqueta híbrida), MUCH
# (claim-level, en/es, solo Llama 3.2 3B y Gemma 3 4B) y Mu-SHROOM-ES (span-level,
# zero-shot, generaciones ajenas). MUCH y Mu-SHROOM ya vienen etiquetados (no usan
# juez). `zero_shot` marca los benchmarks de solo prueba (sin folds propios): se
# evalúan con clasificadores entrenados en `train_on`.
BENCHMARKS = {
    "halueval-qa": BenchmarkSpec("halueval-qa", "en", 1000),
    "halueval-qa-gen": BenchmarkSpec("halueval-qa-gen", "en", None),
    "much-en": BenchmarkSpec("much-en", "en", None),
    "much-es": BenchmarkSpec("much-es", "es", None),
    "mushroom-es": BenchmarkSpec("mushroom-es", "es", None),
    # SQuAD: control de reproducción de TOHA (open-book, contexto rico, Fase 4.9).
    "squad": BenchmarkSpec("squad", "en", 800),
    # XQuAD: comprensión lectora paralela EN/ES (mismas preguntas traducidas), para
    # la comparación cross-language CON contexto (Fase 4). 1190 ítems por idioma.
    "xquad-en": BenchmarkSpec("xquad-en", "en", None),
    "xquad-es": BenchmarkSpec("xquad-es", "es", None),
    # MEMERAG-es: RAG nativo en español con etiqueta humana de fidelidad (Fase 5).
    # Detección sobre respuestas dadas (estilo RAGTruth), el régimen de diseño de TOHA.
    # t_max=1152 para encajar el contexto (top-4 pasajes) sin OOM en el M4 de 16 GB.
    "memerag-es": BenchmarkSpec("memerag-es", "es", None, t_max=1152),
    "memerag-es-majority": BenchmarkSpec("memerag-es-majority", "es", None, t_max=1152),
    # MEMERAG-en: contraparte inglesa para la transferencia cross-language en RAG (Fase 5).
    "memerag-en": BenchmarkSpec("memerag-en", "en", None, t_max=1152),
    "memerag-en-majority": BenchmarkSpec("memerag-en-majority", "en", None, t_max=1152),
}

# Benchmarks de solo prueba (zero-shot): clasificador entrenado en otro benchmark
# del mismo modelo e idioma, aplicado sin reajuste (Cap. 5 §5.6).
ZERO_SHOT = {"mushroom-es": "much-es"}

# Modelos que cubre MUCH (reutiliza sus generaciones y anotaciones). El resto de
# benchmarks aplica a los tres modelos; MUCH y Mu-SHROOM, solo a estos dos.
MUCH_MODELS = ("llama32-3b", "gemma3-4b")

# --- TOHA (Cap. 5 §5.4 / diseño §7) ----------------------------------------
TOHA_PROBE_SET_SIZE = 50
TOHA_N_OPT_CANDIDATES = (2, 4, 6, 8, 10)

# --- LapEigvals (Cap. 5 §5.4 / Binkowski et al. 2025) ----------------------
# k autovalores mayores por cabeza; PCA a 512 dims antes de la regresión logística.
LAPEIGVALS_K = 10
LAPEIGVALS_PCA = 512

# --- HalluZig (Cap. 5 §5.1.4 / Samaga et al. 2026) -------------------------
# Grafo por capa: media de cabezas + umbral por percentil k. Persistencia zigzag
# (Dionysus 2). Vectorización: imagen de persistencia 20x20 + entropía + curva de
# Betti (50 pts) en H0 y H1 -> vector fijo de 902 dims, independiente del modelo.
# Clasificador Random Forest de 200 árboles. El sigma de la imagen de persistencia
# se fija constante (no se calibra por fold) sobre el eje normalizado a [0,1].
HALLUZIG_PERCENTILE = 90.0
HALLUZIG_PI_RES = 20
HALLUZIG_PI_SIGMA = 0.1
HALLUZIG_BETTI_POINTS = 50
HALLUZIG_RF_TREES = 200

# --- SEPs (Cap. 5 §5.4 / sonda lineal de estados ocultos) ------------------
# Sonda lineal (StandardScaler + LogisticRegression) sobre el hidden del último
# token, una por capa; la capa óptima se elige por validación interna (holdout
# estratificado) sin ver el fold de test. Variante directa hidden->etiqueta
# (familia INSIDE/SAPLMA), no el SEPs con entropía semántica (ver memoria §5.4).
SEPS_VAL_FRAC = 0.25
SEPS_LR_MAX_ITER = 1000

# --- SelfCheckGPT (Cap. 5 §5.4 / Manakul et al. 2023) ----------------------
# Acotado (diseño §2): subconjunto, N muestreos extra y consistencia por BERTScore.
# El modelo BERTScore es multilingüe (EN+ES) para reutilizarse en MUCH/Mu-SHROOM;
# se usa F1 sin rescale de baseline (para AUROC basta el orden). Las muestras se
# generan con temperatura alta (diversidad, Manakul et al.); la respuesta principal
# del benchmark se generó con GEN_TEMPERATURE.
SELFCHECK_N = 5
SELFCHECK_SUBSET = 300
SELFCHECK_TEMPERATURE = 1.0
BERTSCORE_MODEL = "bert-base-multilingual-cased"
BERTSCORE_NUM_LAYERS = 9

# --- Evaluación ------------------------------------------------------------
N_FOLDS = 5
# Intervalos de confianza por bootstrap (Cap. 5 §5.5.1 / diseño §7). Importan
# especialmente en MUCH, de clase positiva escasa (~4%), donde la media por fold
# es ruidosa y la cifra honesta es el AUROC/AUPRC agrupado con su CI.
N_BOOTSTRAP = 2000
BOOTSTRAP_CI = 0.95

# --- Generación de respuestas + etiquetado (HaluEval-QA generado) ----------
# Generación closed-book concisa (valida: respuestas QA cortas, sin confound de
# longitud por corrección). El etiquetado es híbrido: exact-match + juez Gemini.
QA_INSTRUCTION = "Answer the question with a short, direct answer of a few words only, without explanation."
QA_GEN_MAX_NEW_TOKENS = 48
# El free tier limita peticiones POR DÍA y por modelo (p. ej. gemini-2.5-flash: 20/día).
# flash-lite suele tener cuota mayor; lotes grandes minimizan el nº de llamadas.
GEMINI_JUDGE_MODEL = "gemini-2.5-flash-lite"
JUDGE_BATCH_SIZE = 50          # ítems por llamada al juez (1000 muestras -> ~14 llamadas)
JUDGE_SLEEP_SECONDS = 4.0      # espaciado entre llamadas (solo necesario con Gemini)

# Juez local (Ollama en la VM del trabajo, vía túnel). Se adopta como juez único del
# trabajo tras agotar el free tier de Gemini (cuyo prepago mínimo de $10 queda fuera de
# presupuesto): es gratis, sin cuotas y reutilizable para todo el etiquetado. En el test
# de calibración, mistral-small:24b acertó 12/12 a ~0,38 s/ítem con salida forzada por
# esquema JSON (necesaria: sin esquema los modelos pequeños devuelven JSON degenerado).
JUDGE_BACKEND = "ollama"  # "ollama" (local, por defecto) | "gemini"
OLLAMA_URL = "http://172.17.28.131:11434/api/generate"
OLLAMA_JUDGE_MODEL = "mistral-small:24b"
OLLAMA_TIMEOUT = 300


def load_env_var(name: str) -> str | None:
    """Lee una variable de `.env` en la raíz (p. ej. GEMINI_KEY), sin depender de
    cargarla en el entorno. Cae a os.environ si no está en el fichero."""
    env = PROJECT_ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get(name)
