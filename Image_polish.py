from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
# load SDXL refiner as an img2img pipeline
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    safety_checker = None
).to("cpu")  # or "cuda" if you have GPU

image = Image.open("output.png").convert("RGB").resize((1024,1024))

prompt = "a natural grass background with photorealistic details"

# do a mild denoising
output = pipe(
    prompt=prompt,
    image=image,
    strength=0.2,  # keep close to original
    guidance_scale=7.5
).images[0]

output.save("polished.png")
