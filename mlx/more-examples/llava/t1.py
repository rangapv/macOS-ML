#!/usr/bin/env python3
#author:rangap@yahoo.com
#26-04-26

from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

messages = [
    {
        "role": "user", "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
            {"type": "text", "text": "Where is the cat standing?"},
        ]
    },
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True
)
# Generate
generate_ids = model.generate(**inputs)
result = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
print(result)
