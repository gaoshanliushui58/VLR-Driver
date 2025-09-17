from PIL import Image 
from transformers import AutoModelForCausalLM, AutoProcessor 
import torch

model_path = "Bench2Drive/LMDrive-b2d/phi_vision/Phi-3.5-vision-instruct"
model_id = "microsoft/Phi-3.5-vision-instruct" 


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AutoModelForCausalLM.from_pretrained(
  model_path, 
  #device_map="cuda", 
  trust_remote_code=True, 
  torch_dtype="auto", 
  _attn_implementation='eager' # 'flash_attention_2'    
).to(device)

processor = AutoProcessor.from_pretrained(model_path, 
  trust_remote_code=True, 
  num_crops=4  # 用于多帧任务的数量
)

def generate_response(messages, images):
    prompt = processor.tokenizer.apply_chat_template(
      messages, 
      tokenize=False, 
      add_generation_prompt=True
    )
 
    inputs = processor(prompt, images, return_tensors="pt").to("cuda:0") 
 
    generation_args = { 
        "max_new_tokens": 500,
        "temperature": 0.7,
        "do_sample": True,
        "top_k": 50,
    } 
 
    generate_ids = model.generate(**inputs, 
      eos_token_id=processor.tokenizer.eos_token_id, 
      **generation_args
    )
 
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, 
      skip_special_tokens=True, 
      clean_up_tokenization_spaces=False)[0] 
 
    return response

# 加载图像
image1 = Image.open("Bench2Drive/MergerIntoSlowTrafficV2_Town12_Route1043_Weather3/camera/rgb_front/00050.jpg")
image2 = Image.open("Bench2Drive/MergerIntoSlowTrafficV2_Town12_Route1043_Weather3/camera/rgb_front/00060.jpg")
image3 = Image.open("Bench2Drive/MergerIntoSlowTrafficV2_Town12_Route1043_Weather3/camera/rgb_front/00070.jpg")
image4 = Image.open("Bench2Drive/MergerIntoSlowTrafficV2_Town12_Route1043_Weather3/camera/rgb_front/00080.jpg")

# 多图像提示
# multi_image_prompt = """
# <|image_1|>
# <|image_2|>
# These two images show different pets. Please describe each image in detail, focusing on the animals, their actions, and their surroundings.
# """

multi_image_prompt = """
<|image_1|>
<|image_2|>
<|image_3|>
<|image_4|>
There is a car driving on the road in the picture. Please describe the environment around it. How is he driving, how will he drive in the future, and why is he driving like this?
"""


conversation = [
    {"role": "user", "content": multi_image_prompt}
]
 
response = generate_response(conversation, [image1, image2, image3, image4])
conversation.append({"role": "assistant", "content": response})
 
# 打印对话
for message in conversation:
    print(f"{message['role'].capitalize()}: {message['content']}\n")