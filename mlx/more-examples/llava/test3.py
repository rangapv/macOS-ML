#!/usr/bin/env python3

from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration

from PIL import Image
import requests, torch

processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained(
    "llava-hf/llava-v1.6-mistral-7b-hf", dtype=torch.float16, device_map="auto"
)

# Load multiple images
#url1 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/llava_next_ocr.png"
#url2 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/llava_next_comparison.png"

url1 = "https://huggingface.co/datasets/huggingface/documentation-images/blob/main/transformers/model_doc/sam-car.png"
url2 = "https://huggingface.co/datasets/huggingface/documentation-images/blob/main/transformers/model_doc/sam-car.png"
#url2 = "https://huggingface.co/datasets/huggingface/documentation-images/blob/main/transformers/model_doc/sam-car-seg.png"

image1 = Image.open(requests.get(url1, stream=True).raw)
image2 = Image.open(requests.get(url2, stream=True).raw)

conversation = [
    {"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": "Compare these two images and describe the differences."}]}
]
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor([image1, image2], prompt, return_tensors="pt").to(model.device)

output = model.generate(**inputs, max_new_tokens=100)
print(processor.decode(output[0], skip_special_tokens=True))
