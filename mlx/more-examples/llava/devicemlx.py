#!/usr/bin/env python3
#author:rangapv@yahoo.com
#30-04-26

import mlx.core as mx
import os
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model1="llava-hf/llava-1.5-7b-hf"
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = LlavaForConditionalGeneration.from_pretrained(model1).to(device)
processor = AutoProcessor.from_pretrained(model1)
url1 = "https://static01.nyt.com/images/2023/07/21/multimedia/21baguettesrex-hbkc/21baguettesrex-hbkc-videoSixteenByNineJumbo1600.jpg"

def mesg1():
   input1 = input("USER:")
   m2 = [
       {
        "role": "user",
        "content": [
            {"type": "image", "url": url1},
            {"type": "text", "text": input1}
        ]
       }
   ]
   m1 = m2
   #print(f"inside mesg1 {m1}")
   return m1

def sample(logits, temperature=0.0):
    print("logist type printing")
    print(type(logits))
    #logits_mx = mx.array(logits.detach().numpy())
    #return mx.argmax(logits_mx, axis=-1)
    if temperature == 0:
        return mx.argmax(mx.array(logits.cpu().detach().numpy()), axis=-1)
        #return mx.argmax(logits_mx, axis=-1)
    else:
        return random.categorical(logits_mx * (1 / temperature))

while True:
 message = mesg1()
 inputs = processor.apply_chat_template(
    message,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    add_generation_prompt=True
    #continue_final_message=True
 ).to(device)
# Generate
 #generate_ids = model.generate(**inputs)
 #input_ids = mx.array(inputs["input_ids"], dtype=mx.int64)
 #pixel_values= mx.array(inputs["pixel_values"])
 input_ids = inputs["input_ids"]
 pixel_values= inputs["pixel_values"]
 temperature = 0.0
# logits, cache = model(input_ids=input_ids, pixel_values=pixel_values, return_dict=True)
 output = model(input_ids=input_ids, pixel_values=pixel_values, return_dict=True)
 logits = output[0]
 cache = output[1]
 logits = logits[:, -1, :]
 y = sample(logits, temperature=temperature)
 tokens = [y.item()]
 max_tokens = 200

 #for n in range(max_tokens - 1):
 #       logits, cache = model.language_model(y[None], cache=cache)
 #       logits = logits[:, -1, :]
 #       y = sample(logits, temperature)
 #       token = y.item()
 #       if token == processor.tokenizer.eos_token_id:
 #           break
 #       tokens.append(token)
 #result3 = processor.tokenizer.decode(tokens)

 generate_ids = model.generate(**inputs,max_new_tokens=200)
 result = processor.decode(generate_ids, skip_special_tokens=True)
 #print(f"result is {result}")
 result1 = result[0]
 #print(f"result1 is {result1}")
 w1 = "ASSISTANT:"
 indx = result1.find(w1)
 result2 = result1[indx:]
 print(result2)
