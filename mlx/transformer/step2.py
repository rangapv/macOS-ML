#!/usr/bin/env python3
#author:rangapv@yahoo.com
#date:24-10-25

import torch
from huggingface_hub import login
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="google/gemma-2-2b",
    device="mps")

text = "the secret to baking a really good cake is,"
outputs = pipe(text, max_new_tokens=256)
response = outputs[0]["generated_text"]
print(response)
