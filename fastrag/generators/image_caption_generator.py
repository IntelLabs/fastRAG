from transformers.image_utils import load_image
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM
import torch

def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.xpu.is_available():
        return "xpu"
    return "cpu"

class ImageCaptionGenerator:
    def __init__(self, model_name_or_path = "microsoft/caption-large"):
        print("Loading Image Caption model ...")
        self.model_name_or_path = model_name_or_path
        self.device = get_default_device()
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
    
    def caption(self, path):
        raise NotImplementedError()

class GitImageCaptionGenerator(ImageCaptionGenerator):
    def __init__(self, model_name_or_path = "microsoft/git-large-r-textcaps"):
        super().__init__(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(self.device)

    def caption(self, path):
        image = Image.open(path)

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values=pixel_values.to(self.model.device), max_length=30, do_sample=False)
        generated_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_caption

class VLLMImageCaptionGenerator(ImageCaptionGenerator):
    def __init__(self, model_name_or_path="HuggingFaceTB/SmolVLM-Instruct"):
        super().__init__(model_name_or_path)
        
        self.model = AutoModelForVision2Seq.from_pretrained(model_name_or_path,
                                    torch_dtype=torch.bfloat16,
                                    _attn_implementation="flash_attention_2" if self.device == "cuda" else "eager")

    def caption(self, path):
        # Load images
        image1 = load_image(path)
        
        # Create input messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Can you describe the image?"}
                ]
            },
        ]
        
        # Prepare inputs
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image1], return_tensors="pt")
        inputs = inputs.to(self.model.device)
        # Generate outputs
        generated_ids = self.model.generate(**inputs, max_new_tokens=20, do_sample=False)
        return self.processor.batch_decode(generated_ids[:,inputs["input_ids"].shape[1]:])[0]
