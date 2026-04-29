#!/usr/bin/env python3
#author:rangapv@yahoo.com
#27-04-26

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

model1="llava-hf/llava-1.5-7b-hf"
device = torch.device("cuda" if torch.backends.cuda.is_available() else "cpu")

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
 generate_ids = model.generate(**inputs,max_new_tokens=200)
 result = processor.decode(generate_ids, skip_special_tokens=True)
 #print(f"result is {result}")
 result1 = result[0]
 #print(f"result1 is {result1}")
 w1 = "ASSISTANT:"
 indx = result1.find(w1)
 result2 = result1[indx:]
 #print(f"result2 is {result2}")
 print(result2)
