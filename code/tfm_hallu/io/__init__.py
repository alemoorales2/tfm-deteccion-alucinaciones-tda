"""Lectura/escritura del layout de resultados (diseño §6)."""

from .layout import (
    append_metrics,
    load_array,
    load_meta,
    load_samples,
    result_dir,
    save_barcodes,
    save_extraction,
    save_scores,
)

__all__ = [
    "result_dir",
    "save_extraction",
    "save_barcodes",
    "load_array",
    "load_samples",
    "load_meta",
    "save_scores",
    "append_metrics",
]
