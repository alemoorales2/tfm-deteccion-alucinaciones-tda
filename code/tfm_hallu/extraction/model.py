"""Carga de modelos para extracción de atención.

Centraliza la lógica validada en los smoke tests: dispositivo MPS, dtype por
modelo (bfloat16 para Gemma 3, FP16 para el resto), atención eager (obligatoria
para output_attentions) y carga de Gemma 3 en modo texto con su clase
multimodal si AutoModelForCausalLM no basta.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import ATTN_IMPLEMENTATION, ModelSpec


def _device_and_dtype(spec: ModelSpec):
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        return "cpu", torch.float32
    dtype = torch.bfloat16 if spec.dtype == "bfloat16" else torch.float16
    return device, dtype


def load_model(spec: ModelSpec):
    """Devuelve (model, tokenizer, device) listo para forward con atención."""
    device, dtype = _device_and_dtype(spec)
    tok = AutoTokenizer.from_pretrained(spec.hf_id)

    model = None
    if spec.loader == "gemma3_text":
        try:
            model = AutoModelForCausalLM.from_pretrained(
                spec.hf_id, dtype=dtype, attn_implementation=ATTN_IMPLEMENTATION
            )
        except Exception:  # noqa: BLE001
            from transformers import Gemma3ForConditionalGeneration

            model = Gemma3ForConditionalGeneration.from_pretrained(
                spec.hf_id, dtype=dtype, attn_implementation=ATTN_IMPLEMENTATION
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            spec.hf_id, dtype=dtype, attn_implementation=ATTN_IMPLEMENTATION
        )

    model.to(device)
    model.eval()
    return model, tok, device
