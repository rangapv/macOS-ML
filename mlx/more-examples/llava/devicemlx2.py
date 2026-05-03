#!/usr/bin/env python3
import torch, requests, io
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

# -------------------------------------------------
# 1️⃣  Model / device
# -------------------------------------------------
model_name = "llava-hf/llava-1.5-7b-hf"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = LlavaForConditionalGeneration.from_pretrained(
    model_name, torch_dtype=torch.float16
).to(device)
processor = AutoProcessor.from_pretrained(model_name)

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
    return f"USER: <image>\n{txt}\nASSISTANT:"

def sample(logits, temperature=0.0):
    """Pure PyTorch sampling — greedy or temperature."""
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)  # (B, 1)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)          # (B, 1)

# -------------------------------------------------
# 4️⃣  Interactive loop
# -------------------------------------------------
try:
    while True:
        prompt = mesg1()

        # Processor → PyTorch tensors on device
        inputs = processor(
            text=prompt,
            images=img,
            return_tensors="pt"
        ).to(device)

        input_ids     = inputs["input_ids"]        # (1, seq_len)
        pixel_values  = inputs["pixel_values"]     # (1, C, H, W)
        attention_mask = inputs["attention_mask"]  # (1, seq_len)

        # First forward pass — prompt + image
        with torch.inference_mode():
            out = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                use_cache=True,
            )

        logits = out.logits[:, -1, :]              # (1, vocab)
        cache  = out.past_key_values
        y      = sample(logits, temperature=0.0)   # (1, 1)
        tokens = [int(y.item())]

        # Token-by-token sampling loop
        with torch.inference_mode():
            for _ in range(199):                   # max_new - 1
                # Grow attention mask by 1
                attention_mask = torch.cat(
                    [attention_mask,
                     torch.ones((1, 1), dtype=attention_mask.dtype, device=device)],
                    dim=1
                )
                out = model(
                    input_ids=y,                   # (1, 1)
                    past_key_values=cache,
                    attention_mask=attention_mask,
                    use_cache=True,
                )
                cache  = out.past_key_values
                logits = out.logits[:, -1, :]
                y      = sample(logits, temperature=0.0)
                token  = int(y.item())
                if token == processor.tokenizer.eos_token_id:
                    break
                tokens.append(token)

        result = processor.tokenizer.decode(tokens, skip_special_tokens=True)
        print("\nASSISTANT:", result.strip(), "\n")

except KeyboardInterrupt:
    print("\nBye!")
