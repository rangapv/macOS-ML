#!/usr/bin/env python3
#author:rangapv@yahoo.com
#27-04-26


from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

def mesg1():
   input1 = input("USER:")
   m2 = [
       {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://static01.nyt.com/images/2023/07/21/multimedia/21baguettesrex-hbkc/21baguettesrex-hbkc-videoSixteenByNineJumbo1600.jpg"},
            {"type": "text", "text": input1}
        ]   
       }
   ]   
  # m1 = message.append(m2)   
   m1 = m2
   #print(f"inside mesg1 {m1}")
   #message = m1
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
 )
# Generate
 generate_ids = model.generate(**inputs)
 result = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
 print(result)
