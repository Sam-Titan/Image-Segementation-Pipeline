from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image

# Load the SDXL inpainting pipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    safety_checker = None
).to("cpu")  # needs a decent GPU

# Load your images
image = Image.open("Input_Image.jpg").convert("RGB").resize((1024,1024))
mask_image = Image.open("mask.png").convert("RGB").resize((1024,1024))

prompt = ""

result = pipe(
    prompt=prompt,
    image=image,
    mask_image=mask_image,
).images[0]

result.save("output.png")
