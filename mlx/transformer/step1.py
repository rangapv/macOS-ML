#!/usr/bin/env python3
#author:rangapv@yahoo.com
#date:24-10-25

from transformers import pipeline, infer_device

device = infer_device()

pipeline = pipeline(task="text-generation", model="google/gemma-2-2b", device=device)
pipeline(["the secret to baking a really good cake is ", "a baguette is "])
