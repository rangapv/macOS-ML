#!/usr/bin/env python3
#author:rangapv@yahoo.com
#27-04-26

from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
#model1="liuhaotian/llava-v1.6-vicuna-7b"
model1="llava-hf/llava-1.5-7b-hf"
#model1="llava-hf/llava-onevision-qwen2-0.5b-ov-hf"
model = LlavaForConditionalGeneration.from_pretrained(model1)
processor = AutoProcessor.from_pretrained(model1)
file1 = Image.open("/Users/rangaswamypv/rangapv/macOD-ML/mlx/tensorflow/daisy.jpg")

def mesg1():
   input1 = input("USER:")
   m2 = [
       {
        "role": "user",
        "content": [
            {"type": "image", "image": file1},
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
 )
# Generate
 #generate_ids = model.generate(**inputs)
 generate_ids = model.generate(**inputs,max_new_tokens=625)
 result = processor.decode(generate_ids, skip_special_tokens=True)
 #print(f"result is {result}")
 result1 = result[0]
 #print(f"result1 is {result1}")
 w1 = "ASSISTANT:"
 indx = result1.find(w1)
 result2 = result1[indx:]
 #print(f"result2 is {result2}")
 print(result2)
