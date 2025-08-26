from transformers import SamModel, SamProcessor
from PIL import Image
import torch

# Load the SAM model and processor
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
model = SamModel.from_pretrained("facebook/sam-vit-base").to("cpu")

# Load your image
image = Image.open("Input_Image.jpg").convert("RGB")

# Provide a click prompt, say roughly in the center of the object
input_boxes = [[[240, 340, 400, 600]]]  # (x,y) coordinate

# Process input
inputs = processor(image, input_boxes=input_boxes, return_tensors="pt").to("cpu")

# Get the mask
with torch.no_grad():
    outputs = model(**inputs)

# Post-process masks
masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(),
    inputs["original_sizes"].cpu(),
    inputs["reshaped_input_sizes"].cpu()
)

# Check mask shape
print("masks is a list with length:", len(masks))
mask = masks[0]  # first element
print("mask initial shape:", mask.shape)

# Usually SAM returns shape (1, 1, H, W) — but if 3 channels appear, handle that:
# get the final mask shape correctly
if mask.ndim == 4:
    mask = mask[0, 0, :, :]
elif mask.ndim == 3 and mask.shape[0] == 3:
    mask = mask[0, :, :]
elif mask.ndim == 3 and mask.shape[0] == 1:
    mask = mask[0, :, :]
else:
    mask = mask.squeeze()

print("final mask shape:", mask.shape)

# convert to numpy
mask_np = mask.numpy()
mask_image = Image.fromarray((mask_np * 255).astype("uint8"))
mask_image.save("mask.png")
print("mask saved successfully.")
