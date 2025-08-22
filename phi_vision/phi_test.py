from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import argparse

class VisionInstructModel:
    def __init__(self, model_path, local_image_path, torch_dtype='auto'):
        self.model_path = model_path
        self.local_image_path = local_image_path
        self.torch_dtype = torch_dtype
 
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._load_model_and_processor()
 
    def _load_model_and_processor(self):
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=self.torch_dtype,
            _attn_implementation= 'eager' # 'flash_attention_2'
        ).to(self.device)
 
    def _prepare_input(self, prompt, image_path):
        image = Image.open(image_path)
        # image.save("output_image.jpg")
        return self.processor(prompt, image, return_tensors="pt").to(self.device)
 
    def generate_response(self, prompt, max_new_tokens=1000):
        inputs = self._prepare_input(prompt, self.local_image_path)
        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.processor.tokenizer.eos_token_id
        )
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
 
        response = self.processor.batch_decode(
            generate_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        return response
 
    def describe_image(self):
        user_prompt = '<|user|>\n'
        assistant_prompt = '<|assistant|>\n'
        prompt_suffix = "<|end|>\n"
        prompt = f"{user_prompt}<|image_1|>\nDescribe the picture and tell me what can you see{prompt_suffix}{assistant_prompt}"
 
        response = self.generate_response(prompt)
        print("response:", response)
        return response
 
def main(model_path, image_path):
    model = VisionInstructModel(model_path, image_path, torch_dtype='bfloat16')
    model.describe_image()
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run VisionInstructModel to describe an image.")
    parser.add_argument("--model_path", type=str, default="/data01/kfj_data01/HOME/Bench2Drive/LMDrive-b2d/phi_vision/Phi-3.5-vision-instruct", help="Path to the model directory.")
    parser.add_argument("--image_path", type=str, default="/data01/kfj_data01/DATA/Bench2Drive/Bench2Drive/ConstructionObstacleTwoWays_Town12_Route1415_Weather26/camera/rgb_front/00013.jpg", help="Path to the image file.")
    
    args = parser.parse_args()
    main(args.model_path, args.image_path)