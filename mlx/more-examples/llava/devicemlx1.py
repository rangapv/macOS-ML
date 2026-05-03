#!/isr/bin/env python3
#author: rangapv@yahoo.com
#03-05-2026

import mlx.core as mx
import requests, io
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

# -------------------------------------------------
# 1️⃣  Model / device
# -------------------------------------------------
model_name = "llava-hf/llava-1.5-7b-hf"
device = mx.device()                     # mx.gpu() on Apple Silicon, else mx.cpu()

model = LlavaForConditionalGeneration.from_pretrained(
    model_name,
    trust_remote_code=True,
    _framework="mlx",
).to(device)

processor = AutoProcessor.from_pretrained(model_name)

# -------------------------------------------------
# 2️⃣  Load the image once
# -------------------------------------------------
url = "https://static01.nyt.com/images/2023/07/21/multimedia/21baguettesrex-hbkc/21baguettesrex-hbkc-videoSixteenByNineJumbo1600.jpg"
resp = requests.get(url)
img = Image.open(io.BytesIO(resp.content)).convert("RGB")

# -------------------------------------------------
# 3️⃣  Helper functions
# -------------------------------------------------
def mesg1():
    txt = input("USER: ")
    if txt.lower() in {"quit", "exit"}:
        raise KeyboardInterrupt
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": txt},
            ],
        }
    ]

def sample(logits, temperature=0.0):
    if temperature == 0.0:
        return mx.argmax(logits, axis=-1)
    probs = mx.softmax(logits / temperature, axis=-1)
    return mx.random.categorical(probs, 1).squeeze(-1)

# -------------------------------------------------
# 4️⃣  Interactive loop
# -------------------------------------------------
try:
    while True:
        message = mesg1()
        # Processor returns PyTorch tensors → convert to MLX
        inputs_pt = processor.apply_chat_template(
            message,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        inputs = {k: mx.array(v).to(device) for k, v in inputs_pt.items()}

        input_ids = inputs["input_ids"]
        pixel_values = inputs["pixel_values"]
        attention_mask = inputs["attention_mask"]

        # First forward pass (prompt + image)
        out = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            use_cache=True,
        )
        logits = out.logits[:, -1, :]
        cache = out.past_key_values
        y = sample(logits, temperature=0.0)
        tokens = [int(y.item())]

        max_new = 200
        for _ in range(max_new - 1):
            # extend mask
            attention_mask = mx.concatenate(
                [attention_mask, mx.ones((y.shape[0], 1), dtype=y.dtype)], axis=1
            )
            out = model(
                input_ids=y[:, None],
                past_key_values=cache,
                attention_mask=attention_mask,
                use_cache=True,
            )
            cache = out.past_key_values
            logits = out.logits[:, -1, :]
            y = sample(logits, temperature=0.0)
            token = int(y.item())
            if token == processor.tokenizer.eos_token_id:
                break
            tokens.append(token)

        result = processor.decode(tokens, skip_special_tokens=True)
        # Strip a leading "ASSISTANT:" if present
        prefix = "ASSISTANT:"
        if prefix in result:
            result = result[result.find(prefix) + len(prefix):].strip()
        print("\nASSISTANT:", result, "\n")
except KeyboardInterrupt:
    print("\nSession ended.")
