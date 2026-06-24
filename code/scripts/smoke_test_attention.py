"""Smoke test: verifica que un modelo descargado de HuggingFace se carga
con attn_implementation="eager" + output_attentions=True en MPS y devuelve
matrices de atencion de la forma esperada.

Uso:
    python3 code/scripts/smoke_test_attention.py [modelo]

Por defecto prueba microsoft/Phi-3.5-mini-instruct. Se puede pasar otro id
de HuggingFace como argumento.
"""

import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(model_id: str = "microsoft/Phi-3.5-mini-instruct") -> int:
    print(f"=== Smoke test sobre {model_id} ===\n")

    # 1) Eleccion de dispositivo
    if torch.backends.mps.is_available():
        device = "mps"
        dtype = torch.float16
    elif torch.cuda.is_available():
        device = "cuda"
        dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
    print(f"Dispositivo: {device}, dtype: {dtype}")

    # 2) Tokenizer
    print("\nCargando tokenizer...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    print(f"  ok en {time.time() - t0:.1f}s")

    # 3) Modelo en modo eager (obligatorio para output_attentions).
    # Cargamos en CPU primero y movemos al device con .to() para no depender
    # de accelerate. Para los modelos grandes (mas de 7B) habra
    # que instalar accelerate y usar device_map="auto".
    print("\nCargando modelo (eager attention)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        attn_implementation="eager",
    )
    model.to(device)
    model.eval()
    print(f"  ok en {time.time() - t0:.1f}s")
    print(f"  num_hidden_layers={model.config.num_hidden_layers}")
    print(f"  num_attention_heads={model.config.num_attention_heads}")

    # 4) Forward pass con output_attentions
    prompt = "Madrid es la capital de"
    print(f"\nEntrada: '{prompt}'")
    inputs = tok(prompt, return_tensors="pt").to(device)
    seq_len = inputs["input_ids"].shape[1]

    print("Forward pass con output_attentions=True...")
    t0 = time.time()
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    print(f"  ok en {time.time() - t0:.2f}s")

    # 5) Comprobaciones sobre el tensor de atencion
    attns = outputs.attentions
    L = len(attns)
    H = attns[0].shape[1]
    T = attns[0].shape[2]
    print(f"\n--- Tensores de atencion ---")
    print(f"  numero de capas:      {L}")
    print(f"  cabezas por capa:     {H}")
    print(f"  longitud de secuencia: {T} (= {seq_len} tokens)")
    print(f"  shape por capa:       {tuple(attns[0].shape)}")
    print(f"  matrices totales:     {L * H}")

    # 6) Sanity: cada fila de la matriz de atencion debe sumar 1
    sample = attns[0][0, 0].float().cpu()  # capa 0, cabeza 0
    row_sums = sample.sum(dim=-1)
    if torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3):
        print("  filas suman 1   OK")
    else:
        print(f"  filas suman 1   FALLO  (max desv = {(row_sums - 1).abs().max():.4f})")
        return 1

    # 7) Generacion corta para confirmar que el modelo razona
    print("\nGeneracion corta de comprobacion:")
    t0 = time.time()
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    print(f"  ok en {time.time() - t0:.2f}s")
    text = tok.decode(gen_ids[0], skip_special_tokens=True)
    print(f"  '{text}'")

    print("\n=== Smoke test OK ===")
    return 0


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "microsoft/Phi-3.5-mini-instruct"
    sys.exit(main(model))
