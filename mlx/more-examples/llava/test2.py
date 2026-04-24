#!/usr/bin/env python3
#author:rangapv@yahoo.com
#24-04-2026

from generate import load_model, prepare_inputs, generate_text
import os
import mlx.core as mx

os.environ["TOKENIZERS_PARALLELISM"] = "false"

processor, model = load_model("llava-hf/llava-1.5-7b-hf")

max_tokens, temperature = 128, 0.0

prompt = "USER: <image>\nWhat are these?\nASSISTANT:"
image = "/Users/rangaswamypv/rangapv/macOD-ML/mlx/tensorflow/daisy.jpg"
#image = "http://images.cocodataset.org/val2017/000000039769.jpg"
input_ids, pixel_values = prepare_inputs(processor, image, prompt)
input_ids   = mx.array(input_ids,   dtype=mx.int64)   # ← crucial fix
pixel_values = mx.array(pixel_values)

reply = generate_text(
    input_ids, pixel_values, model, processor, max_tokens, temperature
)

print(reply)
