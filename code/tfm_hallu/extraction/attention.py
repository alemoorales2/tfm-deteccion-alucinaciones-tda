"""Tokenización con partición P/R y extracción de la atención.

La secuencia es la concatenación cruda de los tokens del prompt (P) y de la
respuesta (R). La frontera es exacta: se tokenizan por separado y se concatenan
los ids, de modo que `n_prompt` da la partición sin ambigüedad de subpalabras.

Truncado a `t_max`: se conserva siempre la respuesta entera y, del prompt, el
BOS y su cola (que incluye la pregunta), descartando el medio del knowledge.
"""

from dataclasses import dataclass

import torch

from ..data.base import Sample


@dataclass
class Encoded:
    input_ids: torch.Tensor  # [1, T]
    n_prompt: int            # frontera P/R (número de tokens de P)
    n_tok_resp: int
    truncated: bool


def encode_sample(tokenizer, sample: Sample, t_max: int) -> Encoded:
    chat_mode = sample.meta.get("chat")
    if chat_mode:
        # P = prompt con plantilla de chat (como lo vería el modelo), R = respuesta.
        #   chat == True    -> estilo QA de HaluEval generado (instrucción de concisión).
        #   chat == "plain" -> prompt tal cual (MUCH/Mu-SHROOM traen su propio enunciado).
        from .generate import build_qa_messages

        if chat_mode == "plain":
            messages = [{"role": "user", "content": sample.prompt.strip()}]
        else:
            messages = build_qa_messages(sample.prompt)
        templated = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        # normalizar a list[int] (puede venir como BatchEncoding o anidado)
        if hasattr(templated, "keys"):
            templated = templated["input_ids"]
        if templated and isinstance(templated[0], list):
            templated = templated[0]
        p_ids = list(templated)
    else:
        p_ids = tokenizer(sample.prompt, add_special_tokens=True)["input_ids"]
    r_ids = tokenizer(sample.respuesta, add_special_tokens=False)["input_ids"]

    truncated = False
    if len(p_ids) + len(r_ids) > t_max:
        truncated = True
        budget = t_max - len(r_ids)
        if budget >= 2:
            # conservar BOS + cola del prompt (donde está la pregunta)
            p_ids = p_ids[:1] + p_ids[1:][-(budget - 1):]
        elif budget == 1:
            p_ids = p_ids[:1]
        else:
            # respuesta más larga que t_max: caso extremo, se recorta R
            p_ids = p_ids[:1]
            r_ids = r_ids[: max(t_max - 1, 1)]

    ids = p_ids + r_ids
    input_ids = torch.tensor([ids], dtype=torch.long)
    return Encoded(
        input_ids=input_ids,
        n_prompt=len(p_ids),
        n_tok_resp=len(r_ids),
        truncated=truncated,
    )


@torch.no_grad()
def extract_attention(model, encoded: Encoded, device: str):
    """Forward pass con output_attentions y output_hidden_states. Devuelve
    (attn, hidden):

      - attn:   NumPy [L, H, T, T] float32 (CPU), para TOHA/LapEigvals/HalluZig.
      - hidden: NumPy [L+1, d_model] float16 (CPU), último token por capa (SEPs).

    Un único forward alimenta los cuatro métodos basados en internos. Los tensores
    crudos en el dispositivo se liberan al salir; el llamador solo retiene estos
    dos arrays compactos por muestra."""
    from .hidden import last_token_hidden

    input_ids = encoded.input_ids.to(device)
    out = model(
        input_ids=input_ids,
        output_attentions=True,
        output_hidden_states=True,
        use_cache=False,
    )
    # out.attentions: tupla de L tensores [1, H, T, T]
    attn = torch.stack([a[0] for a in out.attentions]).float().cpu().numpy()
    hidden = last_token_hidden(out.hidden_states)
    del out
    return attn, hidden
