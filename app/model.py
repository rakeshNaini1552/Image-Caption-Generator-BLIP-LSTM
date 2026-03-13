from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

def load_model(device):
    # load processor and model from "Salesforce/blip-image-captioning-base"
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.to(device)
    # return (model, processor)
    return model, processor

def generate_caption(image: Image.Image, model, processor, device) -> str:
    # process image
    inputs = processor(image, return_tensors="pt").to(device)
    # generate caption
    generated_ids = model.generate(**inputs, max_length=200)
    # decode and return as string
    caption = processor.decode(generated_ids[0], skip_special_tokens=True)
    return caption