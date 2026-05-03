#!/usr/bin/env python3
#author:rangapv@yahoo.com
#03-05-2026

import torch
import requests, io
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

# -------------------------------------------------
# 1️⃣  Model / device setup
# -------------------------------------------------
model_name = "llava-hf/llava-1.5-7b-hf"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LlavaForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.float16,          # half‑precision saves VRAM
    low_cpu_mem_usage=True,
).to(device)

processor = AutoProcessor.from_pretrained(model_name)

# -------------------------------------------------
# 2️⃣  Load the image once (the same image for every turn)
# -------------------------------------------------
url1 = "https://static01.nyt.com/images/2023/07/21/multimedia/21baguettesrex-hbkc/21baguettesrex-hbkc-videoSixteenByNineJumbo1600.jpg"

# -------------------------------------------------
# 3️⃣  Helper functions
# -------------------------------------------------
def mesg1():
    """Read user text and build the multimodal message."""
    txt = input("USER: ")
    if txt.lower() in {"quit", "exit"}:
        raise KeyboardInterrupt
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": url1},
                {"type": "text", "text": txt},
            ],
        }
    ]

def sample(logits, temperature=0.0):
    """Deterministic argmax when temperature==0, otherwise multinomial."""
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)

# -------------------------------------------------
# 4️⃣  Main interactive loop
# -------------------------------------------------
try:
    while True:
        # ----- Build the prompt -----
        message = mesg1()
        inputs = processor.apply_chat_template(
            message,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(device)

        input_ids = inputs["input_ids"]          # (1, seq_len)
        pixel_values = inputs["pixel_values"]    # (1, 3, H, W)
        attention_mask = inputs["attention_mask"]# (1, seq_len)

        # ----- First forward pass (image + text) -----
        out = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            use_cache=True,
        )
        logits = out.logits[:, -1, :]            # (1, vocab)
        cache = out.past_key_values
        temperature=0.0
        y = sample(logits, temperature)        # (1,)
        tokens = [y.item()]

        # ----- Generation loop -----
        max_new = 200
        for _ in range(max_new - 1):
            # extend attention mask for the newly generated token
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(y[:, None])], dim=1
            )

            out = model(
                input_ids=y[:, None],            # (1, 1)
                past_key_values=cache,
                attention_mask=attention_mask,
                use_cache=True,
            )
            cache = out.past_key_values
            logits = out.logits[:, -1, :]
            y = sample(logits, temperature)
            token = y.item()
            if token == processor.tokenizer.eos_token_id:
                break
            tokens.append(token)

        # ----- Decode & clean up the answer -----
        result = processor.decode(tokens, skip_special_tokens=True)
        # Optional: strip a leading "ASSISTANT:" prefix
        prefix = "ASSISTANT:"
        if prefix in result:
            result = result[result.find(prefix) + len(prefix):].strip()
        print("\nASSISTANT:", result, "\n")
except KeyboardInterrupt:
    print("\nSession ended.")
