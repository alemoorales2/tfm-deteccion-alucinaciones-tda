# Detección de alucinaciones en LLMs mediante Análisis Topológico de Datos

Código experimental del Trabajo Fin de Máster *«Detección de alucinaciones en modelos de
lenguaje mediante Análisis Topológico de Datos»*, del **Máster en Inteligencia Artificial
(UNIR)**.

- **Autores**: Alejandro Morales Miranda y Álvaro José Ramírez Aguilera
- **Director**: Andrés Soto Villaverde

Este repositorio acompaña a la memoria como Anexo A. Reúne el motor experimental completo
(`tfm_hallu`), los *scripts* que orquestan los experimentos y las tablas de resultados
agregadas, de modo que los números y las figuras del trabajo puedan reproducirse.

## Qué hace

El trabajo interpreta las matrices de atención de un modelo de lenguaje como grafos
ponderados y extrae de ellos firmas topológicas mediante homología persistente y persistencia
zigzag. Esas firmas se usan para detectar alucinaciones y se comparan, bajo un protocolo
común, con un representante de cada una de las otras familias de detección. Los cinco
detectores implementados son:

| Método | Familia | Referencia |
|---|---|---|
| **TOHA** | Topológica (homología persistente) | Bazarova et al. (2025) |
| **HalluZig** | Topológica (persistencia zigzag) | Samaga et al. (2026) |
| **LapEigvals** | Espectral (autovalores del laplaciano) | Binkowski et al. (2025) |
| **SEPs** | Estados ocultos (sonda lineal) | Kossen et al. (2024) |
| **SelfCheckGPT** | Caja negra (consistencia entre muestreos) | Manakul et al. (2023) |

Modelos evaluados (abiertos, ejecutables en hardware local): **Llama 3.2 3B**, **Phi-3.5 Mini**
y **Gemma 3 4B**. *Benchmarks*: HaluEval-QA, MUCH (EN/ES), Mu-SHROOM-ES, SQuAD, XQuAD (EN/ES)
y MEMERAG (EN/ES, escenario RAG).

## Estructura del repositorio

```
code/
  tfm_hallu/            Paquete principal (instalable)
    config.py           Constantes del Cap. 5: semilla, rutas, modelos, benchmarks, hiperparámetros
    extraction/         Carga de modelos y extracción de atención y estados ocultos
    topology/           Homología persistente, persistencia zigzag y vectorización de diagramas
    methods/            Los cinco detectores (toha, halluzig, lapeigvals, seps, selfcheckgpt)
    data/               Loaders de cada benchmark
    evaluation/         Validación cruzada, métricas, pooling y transferencia
    io/                 Disposición de ficheros en data/results
    labeling.py         Etiquetado híbrido (exact-match/F1 + juez basado en LLM)
    viz.py              Generación de las figuras del Cap. 6
  scripts/              Orquestadores del pipeline (ver más abajo)
  notebooks/            Cuadernos pedagógicos (fundamentos de TDA, atención -> grafo)
  pyproject.toml
data/
  results/              Tablas agregadas para reanálisis (~2 MB, ver «Datos»)
requirements.txt
```

## Instalación

Requiere Python 3.12 o superior.

```bash
python3 -m venv .venv
source .venv/bin/activate          # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e code/
python code/scripts/00_check_env.py   # verifica el entorno
```

## Reproducción

Hay dos niveles, según se quiera o no volver a ejecutar los modelos.

### Nivel 1 — Reanálisis sin modelos (barato, sin GPU)

Las tablas agregadas de `data/results/` (métricas, *scores* fuera de *fold*, índice de muestras
y metadatos) permiten regenerar el análisis y las figuras sin re-extraer nada:

```bash
python code/scripts/30_aggregate.py     # regenera las métricas a partir de los scores
python code/scripts/42_master_table.py  # tabla maestra de AUROC por benchmark x modelo x método
```

### Nivel 2 — Re-extracción completa (caro: modelos + horas de cómputo)

Los tensores de *features* (`*.pt`) y los *barcodes* de HalluZig se excluyen del repositorio por
tamaño (varios GB) y se regeneran con el *pipeline*. Requiere acceso a los modelos en Hugging
Face (Llama y Gemma exigen aceptar su licencia y un *token*), y los *datasets* (la mayoría vía
`datasets`; MEMERAG se clona de su repositorio oficial). Flujo por experimento:

```bash
# 1) Extracción de atención y estados ocultos (carga el modelo)
python code/scripts/10_extract.py llama32-3b squad

# 2) Evaluación de un método sobre lo extraído (sin modelo)
python code/scripts/20_evaluate.py llama32-3b squad toha
```

Otros *scripts* relevantes: `05_generate*.py` (generación y etiquetado de respuestas),
`11_selfcheck.py` (muestreos de SelfCheckGPT), `22_zeroshot.py` y `23_transfer.py`
(transferencia entre idiomas y modelos), `41_figures_model.py` (figuras que recargan el modelo).

El equipo de referencia fue un Mac M4 de 16 GB con inferencia en MPS; los hiperparámetros de
memoria (`t_max` por modelo y benchmark) están en `code/tfm_hallu/config.py` y reflejan ese
límite.

## Datos

Se versionan solo las tablas pequeñas necesarias para el reanálisis de Nivel 1:

- `data/results/<modelo>/<benchmark>/` — `meta.json` y `samples.parquet` (índice de muestras,
  respuestas generadas y etiquetas).
- `data/results/scores/` — *scores* fuera de *fold* por modelo, benchmark y método.
- `data/results/metrics/` — métricas agregadas (detección, eficiencia, métricas topológicas).

Los *benchmarks* originales y los tensores pesados no se incluyen: los primeros se descargan de
sus fuentes oficiales y los segundos se regeneran con `10_extract.py`.

## Licencia

Código liberado bajo licencia MIT (ver `LICENSE`). Los *datasets* y los modelos empleados se
rigen por sus respectivas licencias.
