"""Smoke test especifico para Gemma 3.

Verifica los cuatro sub-riesgos de usar Gemma 3 como modelo evaluado:
  1. Clase de carga (Gemma 3 4B es multimodal): intenta AutoModelForCausalLM y,
     si falla, Gemma3ForConditionalGeneration en modo solo texto.
  2. Precision: Gemma es sensible a FP16; se usa bfloat16 en MPS/CUDA.
  3. Atencion limpia en eager: output_attentions=True devuelve las matrices
     (1, H, T, T) y las filas suman 1, comprobado en TODAS las capas (locales y
     globales, dado el patron de atencion 5:1 de Gemma 3).
  4. Generacion coherente.

Uso:
    python3 code/scripts/smoke_test_gemma.py [modelo]
Por defecto google/gemma-3-1b-it (solo texto, ligero). Pasar
google/gemma-3-4b-it para validar tambien la carga multimodal en modo texto.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(model_id: str = "google/gemma-3-1b-it") -> int:
    print(f"=== Smoke test Gemma sobre {model_id} ===\n")

    # 1) Dispositivo y dtype (Gemma es sensible a FP16 -> bfloat16)
    if torch.backends.mps.is_available():
        device, dtype = "mps", torch.bfloat16
    elif torch.cuda.is_available():
        device, dtype = "cuda", torch.bfloat16
    else:
        device, dtype = "cpu", torch.float32
    print(f"Dispositivo: {device}, dtype: {dtype}")

    print("\nCargando tokenizer...")
    tok = AutoTokenizer.from_pretrained(model_id)

    # 2) Carga con fallback de clase
    print("\nCargando modelo (eager attention)...")
    t0 = time.time()
    model, loader = None, None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, attn_implementation="eager"
        )
        loader = "AutoModelForCausalLM"
    except Exception as e:  # noqa: BLE001
        print(f"  AutoModelForCausalLM no fue posible: {type(e).__name__}: {e}")
    if model is None:
        try:
            from transformers import Gemma3ForConditionalGeneration

            model = Gemma3ForConditionalGeneration.from_pretrained(
                model_id, dtype=dtype, attn_implementation="eager"
            )
            loader = "Gemma3ForConditionalGeneration (modo texto)"
        except Exception as e:  # noqa: BLE001
            print(f"  Gemma3ForConditionalGeneration fallo: {type(e).__name__}: {e}")
            return 1
    model.to(device)
    model.eval()
    print(f"  ok en {time.time() - t0:.1f}s  via {loader}")

    cfg = model.config
    tcfg = getattr(cfg, "text_config", cfg)
    L = getattr(tcfg, "num_hidden_layers", None)
    H = getattr(tcfg, "num_attention_heads", None)
    print(f"  num_hidden_layers={L}  num_attention_heads={H}")
    print(
        f"  sliding_window={getattr(tcfg, 'sliding_window', None)}  "
        f"pattern={getattr(tcfg, 'sliding_window_pattern', getattr(tcfg, 'layer_types', None))}"
    )

    # 3) Forward con output_attentions
    prompt = "Madrid es la capital de"
    inputs = tok(prompt, return_tensors="pt").to(device)
    T = inputs["input_ids"].shape[1]
    print(f"\nForward con output_attentions=True (T={T} tokens)...")
    t0 = time.time()
    with torch.no_grad():
        out = model(**inputs, output_attentions=True)
    print(f"  ok en {time.time() - t0:.2f}s")

    attns = getattr(out, "attentions", None)
    if attns is None:
        print("  FALLO: output.attentions es None (las matrices no se exponen)")
        return 2
    print(f"  capas con atencion: {len(attns)}")
    print(f"  shape capa 0:       {tuple(attns[0].shape)}")

    # Filas suman 1 en TODAS las capas (locales y globales)
    ok_layers, bad = 0, []
    for li, a in enumerate(attns):
        if a is None:
            bad.append((li, "None"))
            continue
        rs = a[0].float().sum(dim=-1)
        if torch.allclose(rs, torch.ones_like(rs), atol=1e-2):
            ok_layers += 1
        else:
            bad.append((li, f"desv max {(rs - 1).abs().max():.3f}"))
    print(f"  filas suman 1 en:   {ok_layers}/{len(attns)} capas")
    if bad:
        print(f"  capas con problema: {bad[:8]}{' ...' if len(bad) > 8 else ''}")

    # 4) Generacion corta
    print("\nGeneracion corta de comprobacion:")
    t0 = time.time()
    with torch.no_grad():
        gen = model.generate(
            **inputs, max_new_tokens=10, do_sample=False, pad_token_id=tok.eos_token_id
        )
    print(f"  ok en {time.time() - t0:.2f}s")
    print(f"  '{tok.decode(gen[0], skip_special_tokens=True)}'")

    verdict = "OK" if (ok_layers == len(attns) and not bad) else "REVISAR"
    print(f"\n=== Smoke test Gemma {verdict} ===")
    return 0 if verdict == "OK" else 3


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "google/gemma-3-1b-it"
    sys.exit(main(model))
