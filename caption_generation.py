import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load the image captioning model and processor from the pretrained model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# If CUDA is available, use it; otherwise, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def generate_caption(pil_image):
    try:
        inputs = processor(pil_image, return_tensors="pt").to(device)
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"Error in generate_caption: {e}")
        return "Unable to process image."