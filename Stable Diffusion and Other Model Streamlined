import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os

# Load the models
pipe_sd15 = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
pipe_juggernaut = StableDiffusionPipeline.from_pretrained("JuggernautXL", torch_dtype=torch.float16).to("cuda")
pipe_sd_xl_turbo = StableDiffusionPipeline.from_pretrained("StableDiffusionXLTurbo", torch_dtype=torch.float16).to("cuda")

# Function to generate images
def generate_images(prompt, num_images=1):
    images = []
    
    # Generate from Stable Diffusion 1.5
    image_sd15 = pipe_sd15(prompt).images[0]
    images.append(("Stable Diffusion 1.5", image_sd15))
    
    # Generate from Juggernaut XL
    image_juggernaut = pipe_juggernaut(prompt).images[0]
    images.append(("Juggernaut XL", image_juggernaut))
    
    # Generate from Stable Diffusion XL Turbo
    image_sd_xl_turbo = pipe_sd_xl_turbo(prompt).images[0]
    images.append(("Stable Diffusion XL Turbo", image_sd_xl_turbo))
    
    return images

# Function to save the images for selection
def save_images(images, output_dir="output_images"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filepaths = []
    for i, (model_name, img) in enumerate(images):
        filepath = os.path.join(output_dir, f"{model_name.replace(' ', '_')}_{i}.png")
        img.save(filepath)
        filepaths.append(filepath)
    
    return filepaths

# Prompt the user for a prompt
prompt = input("Enter your prompt: ")

# Generate images from all models
images = generate_images(prompt)

# Save the images to a folder for user selection
image_paths = save_images(images)

# Display the image paths to the user
print("Images generated:")
for idx, path in enumerate(image_paths):
    print(f"{idx + 1}: {path}")

# Ask user to select the best image
selection = int(input("Select the best image by entering the number (1/2/3): ")) - 1

# Get the selected image
best_image_path = image_paths[selection]

# Output the best image
print(f"The selected best image is: {best_image_path}")
