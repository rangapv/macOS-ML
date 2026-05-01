#!/usr/bin/env python3
#author:rangapv@yahoo.com
#30-04-26

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

model1="llava-hf/llava-1.5-7b-hf"
device = torch.device("cuda")

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
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


while True:
 message = mesg1()
 #print(f"inside while {message}")
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
 input_ids=inputs["input_ids"]
 pixel_values=inputs["pixel_values"]
 temperature = 0.0
 #logits, cache , _ = model(input_ids, pixel_values)
 outputs1 = model(input_ids, pixel_values)
 logits =outputs1.logits[:, -1, :]
 y = sample(logits, temperature=temperature)
 tokens = [y.item()]
 max_tokens = 200

 for n in range(max_tokens - 1):
        outputs2 = model(y[None])
        #logits, cache = model(y[None], cache=cache)
        logits = outputs2.logits[:, -1, :]
        y = sample(logits, temperature)
        token = y.item()
        if token == processor.tokenizer.eos_token_id:
            break
        tokens.append(token)

 result3 = processor.tokenizer.decode(tokens)
 print(f"result3 is {result3} and of zero si {result3[0]}")
 #generate_ids = model.generate(**inputs,max_new_tokens=200)
# result = processor.decode(generate_ids, skip_special_tokens=True)
 #print(f"result is {result}")
 result1 = result3[0]
 #print(f"result1 is {result1}")
 w1 = "ASSISTANT:"
 indx = result1.find(w1)
 result2 = result1[indx:]
 #print(f"result2 is {result2}")
 print(result2)
