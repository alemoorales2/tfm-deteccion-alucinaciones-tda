"""Generación de respuestas closed-book (para HaluEval-QA generado).

El modelo responde a la pregunta sin el knowledge (closed-book), con una
instrucción de concisión que produce respuestas estilo QA (pocas palabras). Es
lo que valida que no haya confound de longitud por corrección y lo más fiel a lo
que detectan TOHA/LapEigvals (texto autogenerado por el modelo).
"""

import torch

from .. import config


def build_qa_messages(question: str):
    return [{"role": "user", "content": f"{config.QA_INSTRUCTION}\n\nQuestion: {question.strip()}"}]


@torch.no_grad()
def generate_from_content(model, tok, user_content: str, device: str, max_new_tokens: int | None = None,
                          temperature: float | None = None) -> str:
    """Genera la respuesta a un mensaje de usuario ya construido (plantilla de chat).
    Útil cuando el prompt no es solo la pregunta (p. ej. SQuAD: contexto + pregunta).
    El mismo `user_content` se almacena como P en el loader, de modo que la partición
    P/R de la extracción reproduce exactamente lo que vio el modelo al generar."""
    max_new_tokens = max_new_tokens or config.QA_GEN_MAX_NEW_TOKENS
    temperature = config.GEN_TEMPERATURE if temperature is None else temperature
    enc = tok.apply_chat_template(
        [{"role": "user", "content": user_content}],
        add_generation_prompt=True, return_tensors="pt", return_dict=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    n_prompt = enc["input_ids"].shape[1]
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=config.GEN_TOP_P,
        pad_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0, n_prompt:], skip_special_tokens=True).strip()


@torch.no_grad()
def generate_answer(model, tok, question: str, device: str, max_new_tokens: int | None = None,
                    temperature: float | None = None) -> str:
    """Genera la respuesta del modelo a una pregunta (closed-book, plantilla de chat)."""
    msg = build_qa_messages(question)[0]["content"]
    return generate_from_content(model, tok, msg, device, max_new_tokens, temperature)
