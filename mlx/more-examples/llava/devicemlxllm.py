#!/usr/bin/env python3
import mlx.core as mx
import requests, io
from PIL import Image
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# -------------------------------------------------
# 1️⃣  Model / device  (pure MLX, no torch)
# -------------------------------------------------
model_name = "mlx-community/llava-1.5-7b-4bit"
model, processor = load(model_name)
config = load_config(model_name)

# -------------------------------------------------
# 2️⃣  Load image once
# -------------------------------------------------
url = "https://static01.nyt.com/images/2023/07/21/multimedia/21baguettesrex-hbkc/21baguettesrex-hbkc-videoSixteenByNineJumbo1600.jpg"
img = Image.open(io.BytesIO(requests.get(url).content)).convert("RGB")

# -------------------------------------------------
# 3️⃣  Helper functions
# -------------------------------------------------
def mesg1():
    txt = input("USER: ")
    if txt.lower() in {"quit", "exit"}:
        raise KeyboardInterrupt
    return txt

def sample(logits, temperature=0.0):
    """Pure MLX sampling — greedy or temperature."""
    if temperature == 0.0:
        return mx.argmax(logits, axis=-1)          # (B,)
    probs = mx.softmax(logits / temperature, axis=-1)
    return mx.random.categorical(probs)            # (B,)

# -------------------------------------------------
# 4️⃣  Interactive loop
# -------------------------------------------------
try:
    while True:
        txt = mesg1()

        # Build prompt in LLaVA chat format
        prompt = apply_chat_template(
            processor, config, txt, num_images=1
        )

        # Processor → MLX arrays
        inputs = processor(
            text=prompt,
            images=img,
            return_tensors="np"        # numpy first, then → MLX
        )
        input_ids      = mx.array(inputs["input_ids"])       # (1, seq_len)
        pixel_values   = mx.array(inputs["pixel_values"])    # (1, C, H, W)
        attention_mask = mx.array(inputs["attention_mask"])  # (1, seq_len)

        # First forward pass — prompt + image
        out = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            use_cache=True,
        )
        mx.eval(out.logits)                        # force eval before sampling

        logits = out.logits[:, -1, :]              # (1, vocab)
        cache  = out.past_key_values
        y      = sample(logits, temperature=0.0)   # (1,)
        tokens = [int(y[0].item())]

        # Token-by-token sampling loop
        for _ in range(199):                       # max_new - 1
            # Grow attention mask by 1
            attention_mask = mx.concatenate(
                [attention_mask,
                 mx.ones((1, 1), dtype=attention_mask.dtype)],
                axis=1
            )

            out = model(
                input_ids=y[:, None],              # (1, 1)
                past_key_values=cache,
                attention_mask=attention_mask,
                use_cache=True,
            )
            mx.eval(out.logits)                    # force eval each step

            cache  = out.past_key_values
            logits = out.logits[:, -1, :]
            y      = sample(logits, temperature=0.0)
            token  = int(y[0].item())

            if token == processor.tokenizer.eos_token_id:
                break
            tokens.append(token)

        result = processor.tokenizer.decode(tokens, skip_special_tokens=True)
        print("\nASSISTANT:", result.strip(), "\n")

except KeyboardInterrupt:
    print("\nBye!")
