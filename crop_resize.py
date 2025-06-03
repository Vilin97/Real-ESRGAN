#%%
from PIL import Image

#%%
# Open the image
img = Image.open('vas_hr.jpg')

# Get dimensions
width, height = img.size

# Desired new height
new_height = 3000

# Calculate coordinates for cropping the center portion with original width and new height
top = (height - new_height) // 2
bottom = top + new_height
left = 0
right = width

# Crop the center portion
img_cropped = img.crop((left, top, right, bottom))
img_cropped = img_cropped.resize((1024, 1024), Image.LANCZOS)
img_cropped.size
#%%
# Save as PNG
img_cropped.save('train_pico/gt/vas.png')
# %%
img = Image.open('train_pico/gt/vas.png')
img.size

#%%
from PIL import Image, ImageOps
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import os
import shutil

#%%
model_id = "peter-sushko/RealEdit"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None, cache_dir = 'cache')
pipe.to('cuda')
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

#%%
def encode(pipe, image):
    # Load and preprocess the image
    peter_pixels = np.asarray(image).astype(np.float32) / 255.0  # Normalize to [0, 1]
    peter_tensor = torch.tensor(peter_pixels).permute(2, 0, 1).unsqueeze(0)  # Shape: (1, 3, H, W)
    # Move to the same dtype and device as the VAE expects
    peter_tensor = peter_tensor.to(dtype=pipe.vae.dtype, device=pipe.device)
    # Encode
    encoded = pipe.vae.encode(peter_tensor)
    latent = encoded.latent_dist.sample()
    return latent

def decode(pipe, latent):
    decoded = pipe.vae.decode(latent)
    # 1. Get the image tensor (usually in decoded.sample)
    image_tensor = decoded.sample  # This is usually a tensor like (1, 3, H, W)
    # 2. Squeeze batch dimension and move to CPU
    image_tensor = image_tensor.squeeze(0).detach().cpu()
    # 3. Clamp values to [0, 1] to make it valid for display
    image_tensor = image_tensor.clamp(0, 1)
    # 4. Convert to numpy and then PIL image
    image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    image_pil = Image.fromarray(image_np)
    return image_pil

#%%
"Degrade a single image with the VAE"

path = "vas_hr.png"
img = Image.open(path).convert("RGB")
size=256
img = ImageOps.pad(img, (size, size), method=Image.LANCZOS, color=(0, 0, 0))
display(img)

latent = encode(pipe, img)
decoded_img = decode(pipe, latent)

display(decoded_img)
img = img.resize((1024, 1024), Image.LANCZOS)

# other_img = Image.open("train_pico/gt/vas.png").convert("RGB")
# display(img)
# display(other_img)
#%%
path = "vas_hr.png"
img = Image.open(path).convert("RGB")
img = img.resize((1024, 1024), Image.LANCZOS)
img.save("train_pico/gt/vas.png")

decoded_img.save("train_pico/lq/vas.png")
# %%
# Resize images in train_64/lq to 256x256 and train_64/gt to 1024x1024
for split in ['train_64', 'val_8']:
    for subdir, size in [('lq', 256), ('gt', 1024)]:
        dir_path = os.path.join(split, subdir)
        if not os.path.exists(dir_path):
            continue
        for fname in os.listdir(dir_path):
            fpath = os.path.join(dir_path, fname)
            try:
                img = Image.open(fpath).convert("RGB")
                img = img.resize((size, size), Image.LANCZOS)
                img.save(fpath)
            except Exception as e:
                print(f"Error processing {fpath}: {e}")